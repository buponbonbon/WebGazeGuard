import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def list_subjects(f, prefix="p"):
    return sorted([k for k in f.keys() if k.startswith(prefix)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="Path to MPIIFaceGaze.h5")
    ap.add_argument("--subject_prefix", default="p", help="Subject key prefix (default: p)")
    ap.add_argument("--n_samples", type=int, default=200000, help="Random session samples for gaze stats/plots")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--out_dir", default="analysis_outputs", help="Output folder for plots")
    ap.add_argument("--no_plots", action="store_true", help="Disable saving plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.h5, "r") as f:
        subjects = list_subjects(f, prefix=args.subject_prefix)
        if len(subjects) == 0:
            raise SystemExit("No subjects found. Check subject_prefix.")

        ex_subj = subjects[0]
        ex_sess = list(f[ex_subj]["image"].keys())[0]
        ex_img_shape = f[ex_subj]["image"][ex_sess].shape
        ex_gaze_shape = f[ex_subj]["gaze"][ex_sess].shape

        print("===================================")
        print("EXAMPLE SHAPES (first subject/session)")
        print("===================================")
        print("example subject:", ex_subj)
        print("example session:", ex_sess)
        print("image shape:", ex_img_shape)
        print("gaze shape: ", ex_gaze_shape)

        # Structure scan
        total_sessions = 0
        total_frames_img = 0
        sessions_per_subject = []
        frames_per_session = []

        gaze_dim_set = set()
        gaze_nan_sessions = 0
        gaze_inf_sessions = 0

        for subj in subjects:
            if "image" not in f[subj] or "gaze" not in f[subj]:
                continue

            sessions = list(f[subj]["image"].keys())
            sessions_per_subject.append(len(sessions))
            total_sessions += len(sessions)

            for s in sessions:
                imgs = f[subj]["image"][s]
                Ti = int(imgs.shape[0])
                total_frames_img += Ti
                frames_per_session.append(Ti)

                g = np.asarray(f[subj]["gaze"][s][:], dtype=np.float32).reshape(-1)
                gaze_dim_set.add(int(g.shape[0]))
                if np.isnan(g).any():
                    gaze_nan_sessions += 1
                if np.isinf(g).any():
                    gaze_inf_sessions += 1

        sps = np.array(sessions_per_subject, dtype=np.int64) if sessions_per_subject else np.array([0])
        fps = np.array(frames_per_session, dtype=np.int64) if frames_per_session else np.array([0])

        gaze_dims = sorted(list(gaze_dim_set))
        gaze_dim_str = ", ".join(str(d) for d in gaze_dims) if gaze_dims else "unknown"

        print("\n===================================")
        print("DATASET STRUCTURE SUMMARY")
        print("===================================")
        print("TOTAL SUBJECTS:", len(subjects))
        print("TOTAL SESSIONS:", int(total_sessions))
        print("TOTAL FRAMES (image):", int(total_frames_img))
        print("FRAMES PER SESSION (image) min/max/avg:", int(fps.min()), int(fps.max()), float(fps.mean()))
        print("SESSIONS PER SUBJECT min/max/avg:", int(sps.min()), int(sps.max()), float(sps.mean()))
        print("GAZE VECTOR DIM (per session):", gaze_dim_str)
        print("SESSIONS with NaN in gaze:", int(gaze_nan_sessions))
        print("SESSIONS with Inf in gaze:", int(gaze_inf_sessions))

        # Sample gaze vectors (per session)
        print("\n===================================")
        print("GAZE LABEL STATS (SAMPLED SESSIONS)")
        print("===================================")

        sess_cache = {subj: list(f[subj]["gaze"].keys()) for subj in subjects}
        gaze_list = []
        for _ in range(args.n_samples):
            subj = rng.choice(subjects)
            sessions = sess_cache[subj]
            if not sessions:
                continue
            sess = rng.choice(sessions)
            g = np.asarray(f[subj]["gaze"][sess][:], dtype=np.float32).reshape(-1)
            gaze_list.append(g)

        if len(gaze_list) == 0:
            raise SystemExit("No gaze samples collected (unexpected).")

        gaze = np.stack(gaze_list, axis=0)
        print("Sampled gaze shape:", gaze.shape)

        nan_count = int(np.isnan(gaze).sum())
        inf_count = int(np.isinf(gaze).sum())
        print("NaN count (elements):", nan_count)
        print("Inf count (elements):", inf_count)

        dims = gaze.shape[1]
        names = ["yaw", "pitch"] if dims >= 2 else [f"dim{i}" for i in range(dims)]

        for j in range(min(2, dims)):
            x = gaze[:, j]
            p1, p5, p50, p95, p99 = np.percentile(x, [1, 5, 50, 95, 99])
            print(f"\n== {names[j]} stats ==")
            print("min/max:", float(x.min()), float(x.max()))
            print("mean/std:", float(x.mean()), float(x.std()))
            print("p1/p5/p50/p95/p99:", [float(p1), float(p5), float(p50), float(p95), float(p99)])

        if args.no_plots:
            return

        # Save plots
        if dims >= 1:
            plt.figure()
            plt.hist(gaze[:, 0], bins=100)
            plt.title("Histogram yaw (sampled sessions)")
            plt.xlabel("yaw")
            plt.ylabel("count")
            plt.savefig(os.path.join(args.out_dir, "hist_yaw.png"), dpi=150, bbox_inches="tight")
            plt.close()

        if dims >= 2:
            plt.figure()
            plt.hist(gaze[:, 1], bins=100)
            plt.title("Histogram pitch (sampled sessions)")
            plt.xlabel("pitch")
            plt.ylabel("count")
            plt.savefig(os.path.join(args.out_dir, "hist_pitch.png"), dpi=150, bbox_inches="tight")
            plt.close()

            plt.figure()
            plt.scatter(gaze[:, 0], gaze[:, 1], s=1)
            plt.title("Yaw vs Pitch (sampled sessions)")
            plt.xlabel("yaw")
            plt.ylabel("pitch")
            plt.savefig(os.path.join(args.out_dir, "scatter_yaw_pitch.png"), dpi=150, bbox_inches="tight")
            plt.close()

        print("\nSaved plots to:", os.path.abspath(args.out_dir))

if __name__ == "__main__":
    main()
