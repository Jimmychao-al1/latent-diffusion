import argparse, os, sys, glob, datetime, yaml, shutil, random
import torch
import time
import numpy as np
from tqdm import trange
import lmdb
from io import BytesIO
from torchvision import transforms
import torchvision

from omegaconf import OmegaConf
from PIL import Image

def _bootstrap_local_paths():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidate_paths = [
        repo_root,
        os.path.join(repo_root, "src", "taming-transformers"),
        os.path.join(repo_root, "src", "clip"),
    ]
    for path in candidate_paths:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

_bootstrap_local_paths()

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

rescale = lambda x: (x + 1.) / 2.

def __seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0,):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir, '*.png')))
    # path = logdir
    if model.cond_stage_model is None:
        all_images = []

        print(f"Running unconditional sampling for {n_samples} samples")
        total_batches = (max(n_samples - n_saved, 0) + batch_size - 1) // batch_size
        for _ in trange(total_batches, desc="Sampling Batches (unconditional)"):
            current_batch_size = min(batch_size, n_samples - n_saved)
            if current_batch_size <= 0:
                break
            logs = make_convolutional_sample(model, batch_size=current_batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
        if nplog is not None:
            all_img = np.concatenate(all_images, axis=0)
            all_img = all_img[:n_samples]
            shape_str = "x".join([str(x) for x in all_img.shape])
            nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
            np.savez(nppath, all_img)

    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def build_output_dirs(outputs_root, dataset_tag):
    dataset_root = os.path.join(outputs_root, dataset_tag)
    gen_dir = os.path.join(dataset_root, "gen_images")
    eval_dir = os.path.join(dataset_root, "eval_images")
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    return gen_dir, eval_dir


def reset_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def export_real_from_lmdb(lmdb_path, eval_dir, num_images, img_size, lmdb_resolution, lmdb_zfill):
    existing_pngs = glob.glob(os.path.join(eval_dir, "*.png")) if os.path.exists(eval_dir) else []
    if os.path.exists(eval_dir) and len(existing_pngs) < num_images:
        print(
            f"Rebuilding real image cache: found {len(existing_pngs)} PNGs in {eval_dir}, "
            f"expected at least {num_images}"
        )
        shutil.rmtree(eval_dir)

    os.makedirs(eval_dir, exist_ok=True)
    existing_pngs = glob.glob(os.path.join(eval_dir, "*.png"))
    if len(existing_pngs) >= num_images:
        print(f"Skipping real image export: found {len(existing_pngs)} PNGs in {eval_dir}")
        return

    resize = transforms.Resize(img_size)
    crop = transforms.CenterCrop(img_size)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    env = lmdb.open(
        lmdb_path,
        max_readers=32,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    with env.begin(write=False) as txn:
        length = int(txn.get(b"length").decode("utf-8"))
        if num_images > length:
            raise ValueError(
                f"Requested num_images={num_images}, but LMDB only has length={length}"
            )

        for index in trange(num_images, desc="Exporting real images"):
            key = f"{lmdb_resolution}-{str(index).zfill(lmdb_zfill)}".encode("utf-8")
            img_bytes = txn.get(key)
            if img_bytes is None:
                raise KeyError(f"Missing LMDB key: {key!r}")
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            img = resize(img)
            img = crop(img)
            # Match diffae loader_to_path flow: normalized tensor -> denormalize -> save_image.
            tensor = to_tensor(img)
            tensor = normalize(tensor)
            tensor = (tensor + 1) / 2
            torchvision.utils.save_image(tensor, os.path.join(eval_dir, f"{index}.png"))

    env.close()


def compute_fid(real_dir, gen_dir, fid_batch_size, device, dims=2048):
    from pytorch_fid import fid_score

    return fid_score.calculate_fid_given_paths(
        [real_dir, gen_dir], fid_batch_size, device=str(device), dims=dims
    )


def format_fid_at(num_samples):
    if num_samples % 1000 == 0:
        return f"{num_samples // 1000}k"
    return str(num_samples)


def append_shared_fid_result(result_path, result):
    import json

    history = []
    if os.path.exists(result_path):
        try:
            with open(result_path, "r") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                history = existing
            elif isinstance(existing, dict):
                history = [existing]
        except Exception:
            history = []
    history.append(result)
    with open(result_path, "w") as f:
        json.dump(history, f, indent=2)


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    parser.add_argument(
        "--no_npz",
        default=False,
        action='store_true',
        help="disable saving samples as npz",
    )
    parser.add_argument(
        "--dataset_tag",
        type=str,
        required=True,
        help='dataset tag (e.g. "ffhq256", "lsun_bedroom256")',
    )
    parser.add_argument(
        "--outputs_root",
        type=str,
        default="outputs",
        help="base output directory",
    )
    parser.add_argument(
        "--eval_fid",
        action="store_true",
        help="export real images from LMDB and compute FID",
    )
    parser.add_argument(
        "--real_lmdb",
        type=str,
        default=None,
        help="path to real image LMDB",
    )
    parser.add_argument(
        "--lmdb_resolution",
        type=int,
        default=256,
        help="original_resolution used in LMDB keys",
    )
    parser.add_argument(
        "--lmdb_zfill",
        type=int,
        default=5,
        help="zero-padding width used in LMDB keys",
    )
    parser.add_argument(
        "--num_fid_samples",
        type=int,
        default=50000,
        help="number of real images to export for FID",
    )
    parser.add_argument(
        "--fid_batch_size",
        type=int,
        default=32,
        help="batch size for pytorch-fid",
    )
    parser.add_argument(
        "--fid_dims",
        type=int,
        default=2048,
        help="feature dimensions for FID",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="target image size when exporting real images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for reproducibility",
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None
    #__seed_all(opt.seed)

    if opt.eval_fid and not opt.real_lmdb:
        raise ValueError("--real_lmdb is required when --eval_fid is set")

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    print(config)

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")
    gen_dir, eval_dir = build_output_dirs(opt.outputs_root, opt.dataset_tag)
    dataset_out_dir = os.path.join(opt.outputs_root, opt.dataset_tag)
    reset_dir(gen_dir)

    print(dataset_out_dir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(dataset_out_dir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)


    run(model, gen_dir, eta=opt.eta,
        vanilla=opt.vanilla_sample,  n_samples=opt.n_samples, custom_steps=opt.custom_steps,
        batch_size=opt.batch_size, nplog=None if opt.no_npz else gen_dir)

    if opt.eval_fid:
        export_real_from_lmdb(
            lmdb_path=opt.real_lmdb,
            eval_dir=eval_dir,
            num_images=opt.num_fid_samples,
            img_size=opt.img_size,
            lmdb_resolution=opt.lmdb_resolution,
            lmdb_zfill=opt.lmdb_zfill,
        )

        real_count = len(glob.glob(os.path.join(eval_dir, "*.png")))
        gen_count = len(glob.glob(os.path.join(gen_dir, "*.png")))
        if real_count < opt.num_fid_samples:
            raise ValueError(
                f"Not enough real images for FID: expected at least {opt.num_fid_samples}, got {real_count}"
            )
        if gen_count == 0:
            raise ValueError("No generated images found for FID computation.")
        if gen_count != opt.n_samples:
            print(
                f"Warning: generated image count ({gen_count}) != requested n_samples ({opt.n_samples}). "
                "FID will use all images currently in gen_dir."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fid = compute_fid(eval_dir, gen_dir, opt.fid_batch_size, device, opt.fid_dims)
        print(f"FID: {fid:.4f}")

        result = {
            "fid": fid,
            "dataset_tag": opt.dataset_tag,
            "model_ckpt": ckpt,
            "fid_at": format_fid_at(opt.num_fid_samples),
            "num_fid_samples": opt.num_fid_samples,
            "real_dir": eval_dir,
            "gen_dir": gen_dir,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        result_path = os.path.join(opt.outputs_root, "fid_result.json")
        append_shared_fid_result(result_path, result)
        print(f"FID result saved to {result_path}")

    print("done.")
