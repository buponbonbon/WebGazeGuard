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
    ap.add_argument("--n_samples", type=int, default=200000, help="Random gaze samples for stats/plots")
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

        total_sessions = 0
        total_frames_img = 0
        total_frames_gaze = 0

        mismatch_sessions = 0
        zero_len_sessions = 0

        sessions_per_subject = []
        frames_per_session_min = []

        # Quick structure scan (metadata only)
        for subj in subjects:
            if "image" not in f[subj] or "gaze" not in f[subj]:
                continue

            sessions = list(f[subj]["image"].keys())
            sessions_per_subject.append(len(sessions))
            total_sessions += len(sessions)

            for s in sessions:
                imgs = f[subj]["image"][s]
                gazes = f[subj]["gaze"][s]

                Ti = int(imgs.shape[0])
                Tg = int(gazes.shape[0])

                total_frames_img += Ti
                total_frames_gaze += Tg

                if Ti == 0 or Tg == 0:
                    zero_len_sessions += 1

                if Ti != Tg:
                    mismatch_sessions += 1

                frames_per_session_min.append(min(Ti, Tg))

        sps = np.array(sessions_per_subject, dtype=np.int64) if sessions_per_subject else np.array([0])
        fps = np.array(frames_per_session_min, dtype=np.int64) if frames_per_session_min else np.array([0])

        print("===================================")
        print("DATASET STRUCTURE SUMMARY")
        print("===================================")
        print("TOTAL SUBJECTS:", len(subjects))
        print("TOTAL SESSIONS:", int(total_sessions))
        print("TOTAL FRAMES (image):", int(total_frames_img))
        print("TOTAL FRAMES (gaze): ", int(total_frames_gaze))
        print("MISMATCH sessions (Ti != Tg):", int(mismatch_sessions))
        print("ZERO-LENGTH sessions:", int(zero_len_sessions))

        print("\n=== SESSIONS PER SUBJECT ===")
        print("min / max / avg:", int(sps.min()), int(sps.max()), float(sps.mean()))

        print("\n=== FRAMES PER SESSION (min(T_img, T_gaze)) ===")
        print("min / max / avg:", int(fps.min()), int(fps.max()), float(fps.mean()))

        # Random sampling gaze labels (avoid loading all)
        print("\n===================================")
        print("GAZE LABEL SANITY CHECK (SAMPLED)")
        print("===================================")

        gaze_list = []
        sess_cache = {subj: list(f[subj]["gaze"].keys()) for subj in subjects}

        for _ in range(args.n_samples):
            subj = rng.choice(subjects)
            sessions = sess_cache[subj]
            if not sessions:
                continue
            sess = rng.choice(sessions)

            gazes = f[subj]["gaze"][sess]
            T = int(gazes.shape[0])
            if T <= 0:
                continue

            idx = int(rng.integers(0, T))
            g = np.asarray(gazes[idx], dtype=np.float32)
            if g.ndim == 0:
                g = np.array([g], dtype=np.float32)
            gaze_list.append(g)

        if len(gaze_list) == 0:
            raise SystemExit("No gaze samples collected (unexpected).")

        gaze = np.stack(gaze_list, axis=0)
        print("Sampled gaze shape:", gaze.shape)

        nan_count = int(np.isnan(gaze).sum())
        inf_count = int(np.isinf(gaze).sum())
        print("NaN count:", nan_count)
        print("Inf count:", inf_count)

        dims = min(2, gaze.shape[1])
        names = ["yaw", "pitch"][:dims]
        for j, name in enumerate(names):
            x = gaze[:, j]
            p1, p5, p50, p95, p99 = np.percentile(x, [1, 5, 50, 95, 99])
            print(f"\n== {name} stats ==")
            print("min/max:", float(x.min()), float(x.max()))
            print("mean/std:", float(x.mean()), float(x.std()))
            print("p1/p5/p50/p95/p99:", [float(p1), float(p5), float(p50), float(p95), float(p99)])

        if args.no_plots:
            return

        if dims >= 1:
            plt.figure()
            plt.hist(gaze[:, 0], bins=100)
            plt.title("Histogram yaw (sampled)")
            plt.xlabel("yaw")
            plt.ylabel("count")
            out1 = os.path.join(args.out_dir, "hist_yaw.png")
            plt.savefig(out1, dpi=150, bbox_inches="tight")
            plt.close()

        if dims >= 2:
            plt.figure()
            plt.hist(gaze[:, 1], bins=100)
            plt.title("Histogram pitch (sampled)")
            plt.xlabel("pitch")
            plt.ylabel("count")
            out2 = os.path.join(args.out_dir, "hist_pitch.png")
            plt.savefig(out2, dpi=150, bbox_inches="tight")
            plt.close()

            plt.figure()
            plt.scatter(gaze[:, 0], gaze[:, 1], s=1)
            plt.title("Yaw vs Pitch (sampled)")
            plt.xlabel("yaw")
            plt.ylabel("pitch")
            out3 = os.path.join(args.out_dir, "scatter_yaw_pitch.png")
            plt.savefig(out3, dpi=150, bbox_inches="tight")
            plt.close()

        print("\nSaved plots to:", os.path.abspath(args.out_dir))

if __name__ == "__main__":
    main()
