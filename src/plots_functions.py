import matplotlib.pyplot as plt


def plot_dropsonde(ds):
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
    ds["ta"].plot(y="gpsalt", ax=axes[0], label="Temperature")
    ds["q"].plot(y="gpsalt", ax=axes[1], label="Specific humidity")
    ds["p"].plot(y="gpsalt", ax=axes[2], label="Pressure")
    axes[0].set_xlabel("Temperature / K")
    axes[1].set_xlabel("Specific humidity")
    axes[2].set_xlabel("Pressure / hPa")
    axes[3].set_xlabel("Altitude / m")
