import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
try:
    font_manager.findfont(
        "Times New Roman",
        fontext="ttf",
        directory=None,
        fallback_to_default=False,
        rebuild_if_missing=False,
    )
    plt.rcParams.update({"font.family": "Times New Roman"})
except:
    try:
        font_manager.findfont(
            "Nimbus Roman",
            fontext="ttf",
            directory=None,
            fallback_to_default=False,
            rebuild_if_missing=False,
        )
        plt.rcParams.update({"font.family": "Nimbus Roman"})
    except:
        plt.rcParams.update({"font.family": "Serif"})
folder = '../data/facades'

plt.rcParams.update({"font.size": 14})
df = pd.read_csv(os.path.join(folder,'all.log'), header = None)
df[0] = df[0].str.replace('Epoch: ','').str.replace('\t Train D Loss:',',').str.replace('\t Train G Loss:',',').str.replace('\t Val D Loss: 0 \t Val G Loss: 0','')
df[['Epochs','Discriminator Loss','Generator Loss']] = df[0].str.split(', ', expand=True)
df = df[['Epochs','Discriminator Loss','Generator Loss']]
df = df.astype('float64')
df[ '10_rolling_avg_D' ] = df['Discriminator Loss'].rolling( 10).mean()
df[ '100_rolling_avg_G' ] = df['Generator Loss'].rolling( 100).mean()


plt.figure(dpi=200)
plt.plot(df['Epochs'], df['Discriminator Loss'],color="blue", alpha=0.3)
# plot using rolling average
plt.plot( df['Epochs'], df['10_rolling_avg_D'],color="blue", alpha=1)
plt.xlabel('Epochs')
plt.ylabel('Discriminator Loss')
plt.savefig(os.path.join(folder,"results", f"Discriminator Loss_plot.png"), dpi=200, bbox_inches = "tight")

plt.figure(dpi=200)
plt.plot(df['Epochs'], df['Generator Loss'],color="blue", alpha=0.3)
# plot using rolling average
plt.plot( df['Epochs'], df['100_rolling_avg_G'],color="blue", alpha=1)
plt.xlabel('Epochs')
plt.ylabel('Generator Loss')
plt.savefig(os.path.join(folder,"results", f"Generator Loss_plot.png"), dpi=200, bbox_inches = "tight")
