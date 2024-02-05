import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imufusion
from scipy import signal


def zupt(
    df,
    fn="data",
    sample_rate=200,
    zupt_tresh=3,
    margin=0.1,
    debug=False,
    lp_filter=False,
):
    plt.style.use("default")

    df = df.copy().reset_index()
    df.index /= sample_rate
    df["gyro"] *= 180 / np.pi
    # df['accel'] /= 9.80665

    dt = 1 / sample_rate
    margin = int(margin * sample_rate)  # 100 ms
    # debug = False

    df.index /= sample_rate

    # filter results
    if lp_filter:
        b, a = signal.butter(10, 20, fs=200, btype="lowpass", analog=False)
        df = df.apply(lambda x: signal.filtfilt(b, a, x))

    offset = imufusion.Offset(sample_rate)
    ahrs = imufusion.Ahrs()
    ahrs.settings = imufusion.Settings(
        imufusion.CONVENTION_NED,
        0.5,  # gain
        2000,  # gyroscope range
        10,  # acceleration rejection
        30,  # magnetic rejection
        5 * sample_rate,
    )  # rejection timeout = 5 seconds

    def update(x):
        # ahrs.update(x['gyro'].to_numpy(), x['accel'].to_numpy(), x['mag'].to_numpy(), 0.005)
        ahrs.update_no_magnetometer(
            x["gyro"].to_numpy(), x["accel"].to_numpy(), 0.005
        )

        euler = ahrs.quaternion.to_euler()
        Q = ahrs.quaternion.wxyz
        # acceleration = ahrs.earth_acceleration * 9.80665  # convert g to m/s/
        acceleration = ahrs.earth_acceleration  # convert g to m/s/

        ans = {}
        ans.update(
            {"x": acceleration[0], "y": acceleration[1], "z": acceleration[2]}
        )
        ans.update({"roll": euler[0], "pitch": euler[1], "yaw": euler[2]})
        ans.update({"Q_T": Q})
        ans.update({"accel_err": ahrs.internal_states.acceleration_error})
        ans.update({"accel_igr": ahrs.internal_states.accelerometer_ignored})
        ans.update(
            {"accel_rec": ahrs.internal_states.acceleration_recovery_trigger}
        )

        ans.update({"ang_rrec": ahrs.flags.angular_rate_recovery})
        ans.update({"accel_rrec": ahrs.flags.acceleration_recovery})
        return ans

    sf = df.apply(update, axis=1)
    sf = pd.DataFrame(list(sf), index=df.index)

    fig, ax = plt.subplots(
        nrows=6,
        sharex=True,
        figsize=(20, 15),
        tight_layout=True,
        gridspec_kw={"height_ratios": [6, 6, 6, 2, 1, 1]},
    )

    ax[0].plot(df.index, df["gyro", "x"], "tab:red", label="gyro x")
    ax[0].plot(df.index, df["gyro", "y"], "tab:green", label="gyro y")
    ax[0].plot(df.index, df["gyro", "z"], "tab:blue", label="gyro z")
    ax[0].set_ylabel("Degrees/s")
    ax[0].set_title("Gyroscope")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(df.index, df["accel", "x"], "tab:red", label="Accelerometer x")
    ax[1].plot(
        df.index, df["accel", "y"], "tab:green", label="Accelerometer y"
    )
    ax[1].plot(
        df.index, df["accel", "z"], "tab:blue", label="Accelerometer z"
    )
    ax[1].set_ylabel("Acceleration [g]")
    ax[1].set_title("Accelerometer")
    ax[1].legend()
    ax[1].grid()

    ax[2].plot(sf["yaw"], "tab:red", label="Yaw")
    ax[2].plot(sf["pitch"], "tab:green", label="Pitch")
    ax[2].plot(sf["roll"], "tab:blue", label="Roll")
    ax[2].set_ylabel("Degrees")
    ax[2].grid()
    ax[2].legend()

    ax[3].plot(sf["accel_err"], "tab:olive", label="Acceleration error")
    ax[3].set_ylabel("Error")
    ax[3].set_xlabel("Time [s]")
    ax[3].legend()

    ax[4].plot(sf["accel_igr"], "tab:cyan", label="Acceleration ignored")
    ax[4].legend()

    ax[5].plot(
        sf["accel_rec"], "tab:orange", label="Acceleration recovery trigger"
    )
    ax[5].legend()

    for axes in ax:
        axes.set_xlim(0, sf.index.max())

    fig.savefig(f"ypr.png", dpi=300)

    from scipy.signal import find_peaks

    hf = sf[["x", "y", "z"]].to_numpy()
    cols = pd.MultiIndex.from_product([["acceleration"], ["x", "y", "z"]])
    hf = pd.DataFrame(hf, columns=pd.MultiIndex.from_tuples(cols))

    # subtract earth gravity
    g_end = np.linalg.norm(hf["acceleration"], axis=1)[-100:].mean()
    g_start = abs(hf["acceleration", "z"][-100:].mean())
    g = min(g_start, g_end)
    print(f"calculated g : {g}")

    # ZUPT
    fig, ax = plt.subplots(
        nrows=4, sharex=True, figsize=(20, 10), tight_layout=True
    )

    hf["is_moving"] = (
        hf["acceleration"].apply(np.linalg.norm, axis=1) > zupt_tresh + g
    )

    ax[0].plot(
        hf["acceleration"].apply(np.linalg.norm, axis=1) - g, label="norm"
    )
    ax[1].plot(hf["is_moving"], label="is_moving")

    for index in range(len(hf) - margin):
        hf.loc[index, "is_moving"] = any(
            hf.loc[index : (index + margin), "is_moving"]
        )  # add leading margin

    ax[2].plot(hf["is_moving"], label="is_moving trailing")

    for index in range(len(hf) - 1, margin, -1):
        hf.loc[index, "is_moving"] = any(
            hf.loc[(index - margin) : index, "is_moving"]
        )  # add trailing margin

    ax[3].plot(hf["is_moving"], label="is_moving leading")

    for axes in ax:
        axes.legend()
    ax[0].set_ylim(0, 10)

    if debug:
        fig.savefig(f"zupt.png", dpi=300)
        for axes in ax:
            l = len(hf)
            axes.set_xlim(l / 2 - l / 10, l / 2 + l / 10)
        fig.savefig(f"zupt_zoom.png", dpi=300)

    peaks, _ = find_peaks(hf["is_moving"].astype(int))
    steps = len(peaks)
    print(f"steps : {steps}")

    # velocity caluclations
    velocity = np.zeros((len(hf), 3))
    cols = pd.MultiIndex.from_product([["velocity"], ["x", "y", "z"]])
    hf[cols] = hf["acceleration"] * dt
    for idx in range(1, len(hf)):
        if hf.loc[idx, "is_moving"][0]:
            velocity[idx] = velocity[idx - 1] + hf.loc[idx, "velocity"]

    hf["velocity"] = velocity

    # velocity drift
    is_moving_diff = hf["is_moving"].astype(int).diff().fillna(0)
    idx_shift_diff = is_moving_diff[is_moving_diff < 0].index
    is_moving_diff[idx_shift_diff] = 0
    is_moving_diff[idx_shift_diff - 1] = 1
    is_moving_diff = is_moving_diff.astype(bool)

    hf["step"] = False
    hf.loc[is_moving_diff, "step"] = True

    cols = pd.MultiIndex.from_product([["velocity_drift"], ["x", "y", "z"]])
    hf[cols] = hf["velocity"].apply(lambda x: x * is_moving_diff)
    idx_to_interp = hf[hf["is_moving"]][
        "is_moving"
    ].index.symmetric_difference(is_moving_diff[is_moving_diff].index)
    hf.loc[idx_to_interp, "velocity_drift"] = np.nan
    hf["velocity_drift"] = hf["velocity_drift"].interpolate()
    hf["velocity"] = hf["velocity"] - hf["velocity_drift"]

    # Calculate pos
    cols = pd.MultiIndex.from_product([["position"], ["x", "y", "z"]])
    hf[cols] = hf["velocity"] * dt
    pos = np.zeros((len(hf), 3))
    for idx in range(1, len(hf)):
        pos[idx] = pos[idx - 1] + hf.loc[idx, "position"]

    hf["position"] = pos

    fig, ax = plt.subplots(
        nrows=5,
        sharex=True,
        figsize=(20, 10),
        tight_layout=True,
        gridspec_kw={"height_ratios": [6, 1, 6, 6, 6]},
    )

    ax[0].plot(hf["acceleration", "x"], "tab:red", label="X")
    ax[0].plot(hf["acceleration", "y"], "tab:green", label="Y")
    ax[0].plot(hf["acceleration", "z"], "tab:blue", label="Z")
    ax[0].set_title("Acceleration")
    ax[0].set_ylabel("m/s/s")
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(hf["is_moving"], "tab:cyan", label="Is moving")
    ax[1].grid()
    ax[1].legend()

    ax[2].plot(hf["velocity", "x"], "tab:red", label="X")
    ax[2].plot(hf["velocity", "y"], "tab:green", label="Y")
    ax[2].plot(hf["velocity", "z"], "tab:blue", label="Z")
    ax[2].set_title("Velocity")
    ax[2].set_ylabel("m/s")
    ax[2].grid()
    ax[2].legend()

    ax[3].plot(hf["velocity_drift", "x"], "tab:red", label="X")
    ax[3].plot(hf["velocity_drift", "y"], "tab:green", label="Y")
    ax[3].plot(hf["velocity_drift", "z"], "tab:blue", label="Z")
    ax[3].set_title("Velocity Drift")
    ax[3].set_ylabel("m/s")
    ax[3].grid()
    ax[3].legend()

    ax[4].plot(hf["position", "x"], "tab:red", label="X")
    ax[4].plot(hf["position", "y"], "tab:green", label="Y")
    ax[4].plot(hf["position", "z"], "tab:blue", label="Z")
    ax[4].set_title("Position")
    ax[4].set_ylabel("m")
    ax[4].grid()
    ax[4].legend()

    fig.savefig(f"path_{len(peaks)}.png", dpi=300)

    # plot position 2D

    fig, axes = plt.subplots(nrows=1, figsize=(10, 10))

    axes.plot(hf["position", "x"], hf["position", "y"], label="path")
    axes.scatter(
        hf.loc[idx_shift_diff, "position"]["x"],
        hf.loc[idx_shift_diff, "position"]["y"],
        color="red",
        label="steps",
    )
    axes.set_xlabel("X [m]")
    axes.set_ylabel("Y [m]")
    axes.set_title("position 2D")
    axes.legend()
    axes.grid()

    fig.savefig(f"path2D_{fn}.png", dpi=300)
    plt.show()
