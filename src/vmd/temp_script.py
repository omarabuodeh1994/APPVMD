import seaborn as sns
sns.set()
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x               = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
    y               = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]
    z               = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]

    df = pd.DataFrame(list(zip(x, y, z)), columns =['x', 'y', 'z']) 

    ax = sns.scatterplot(x="x", y="y",
                    hue="z",
                    hue_norm=(0,255),
                    data=df)
    
    plt.xlim(0,255)
    plt.ylim(0,255)
    plt.show()
    
    
    
    
    
    
    