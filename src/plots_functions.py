import matplotlib.pyplot as plt
import numpy as np


def plot_dropsonde(ds_extrap, ds_loc):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    ds_extrap["ta"].plot(y="gpsalt", ax=axes[0, 0], label="Temperature", color="red")
    ds_loc["ta"].plot(y="gpsalt", ax=axes[0, 0], label="Temperature", color="blue")
    ds_extrap["q"].plot(
        y="gpsalt", ax=axes[0, 1], label="Specific humidity", color="red"
    )
    ds_loc["q"].plot(y="gpsalt", ax=axes[0, 1], label="Specific humidity", color="blue")
    ds_extrap["p"].plot(y="gpsalt", ax=axes[0, 2], label="Pressure", color="red")
    ds_loc["p"].plot(y="gpsalt", ax=axes[0, 2], label="Pressure", color="blue")
    ds_extrap["alt"].plot(y="gpsalt", ax=axes[1, 0], label="Altitude", color="red")
    ds_loc["alt"].plot(y="gpsalt", ax=axes[1, 0], label="Altitude", color="blue")
    ds_loc["u"].plot(y="gpsalt", ax=axes[1, 1], label="u", color="blue")
    ds_loc["v"].plot(y="gpsalt", ax=axes[1, 2], label="v", color="blue")
    axes[0, 0].set_xlabel("Temperature / K")
    axes[0, 1].set_xlabel("Specific humidity")
    axes[0, 2].set_xlabel("Pressure / hPa")
    axes[1, 0].set_xlabel("Altitude / m")
    axes[1, 1].set_xlabel("u / m/s")
    axes[1, 2].set_xlabel("v / m/s")

    for ax in axes.flatten():
        ax.set_title("")
        ax.set_ylabel("")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("GPS altitude / m")
    axes[0, 1].set_ylabel("GPS altitude / m")
    plt.show()


def plot_TB_comparison(TB_arts, TB_hamp, sonde_id):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.scatter(TB_hamp.index, TB_hamp, color="blue", marker="o", label="HAMP")
    ax.scatter(TB_arts.index, TB_arts, color="red", marker="x", label="ARTS")
    ax.set_xlabel("Brightness temperature / K")
    ax.set_xticks(TB_arts.index)
    ax.set_xticklabels(TB_arts.index, rotation=45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Frequency / GHz")
    ax.set_title(f"{sonde_id}")
    ax.legend()
    plt.show()


def testplot_hamp(ds_radar, ds_radiometers):
    fig, axes = plt.subplots(
        6, 1, figsize=(10, 20), sharex=True, height_ratios=[3, 1, 1, 1, 1, 1]
    )
    ds_radar["dBZg"].plot.pcolormesh(
        ax=axes[0], x="time", y="height", cmap="viridis", vmin=-30, vmax=30
    )


def plot_regression(regression_coeffs, TB_arts, TB_hamp, date):
    fig, axes = plt.subplots(5, 5, figsize=(15, 15))
    TB_hamp = TB_hamp[TB_hamp.index.date == date]
    TB_arts = TB_arts[TB_arts.index.date == date]
    for i, freq in enumerate(TB_arts.columns):
        x = np.linspace(
            np.nanmin(TB_hamp[freq].values), np.nanmax(TB_hamp[freq].values), 100
        )
        y = (
            regression_coeffs.loc[date, (freq, "slope")] * x
            + regression_coeffs.loc[date, (freq, "intercept")]
        )
        axes.flatten()[i].scatter(
            TB_hamp[freq], TB_arts[freq], color="blue", marker="o", s=1
        )
        axes.flatten()[i].plot(x, y, color="red")
        axes.flatten()[i].set_title(f"{freq} GHz")

    for ax in axes[:, 0]:
        ax.set_ylabel("ARTS TB [K]")
    for ax in axes[-1, :]:
        ax.set_xlabel("HAMP TB [K]")
    fig.tight_layout()
