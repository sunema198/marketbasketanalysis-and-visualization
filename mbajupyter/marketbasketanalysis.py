#%%
from jupyterthemes import jtplot
jtplot.style(theme='onedork', context='talk', fscale=1.4, spines=False, gridlines='--', ticks=True, grid=False, figsize=(6, 4.5))
from os.path import join
import pandas as pd
import numpy as np
import seaborn as sns
current_palette = sns.color_palette()
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib_venn import  venn2
from matplotlib import rcParams
from matplotlib.ticker import FuncFormatter
from scipy.stats import fisher_exact
from ipywidgets import interact, IntSlider, FloatSlider

#%%
total_widget = IntSlider(min=10, max=1000, step=10, value=500)
antecedent_widget = IntSlider(min=5, max=1000, step=5, value=100)
consequent_widget = IntSlider(min=5, max=1000, step=5, value=100)
joint_widget = FloatSlider(min=.01, max=1.0, value=.5)

def plot_metrics(antecedent, consequent, joint_percent, total):
    """Interactive Venn Diagram of joint transactions and plot of support, confidence, and lift  
        Slider Inputs:
            - total: total transactions for all itemsets
            - antecedent, consequent: all transactions involving either itemset
            - joint_percent: percentage of (smaller of) antecedent/consequent involving both itemsets

        Venn Diagram Calculations: 
            - joint = joint_percent * min(antecedent, consequent)
            - antecedent, consequent: original values - joint transactions

        Metric Calculations:
            - Support Antecedent: antecedent/total
            - Support Consequent: Consequent/total
            - Support Joint Transactions: joint/total
            - Rule Confidence: Support Joint Transactions / total
            - Rule Lift: Support Joint Transactions / (Support Antecedent * Support Consequent)
        """

    fig = plt.figure(figsize=(15, 8))
    ax1 = plt.subplot2grid((2, 2), (0, 0)) 
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    ax4 = plt.subplot2grid((2, 2), (1, 1))
 
    
    joint = int(joint_percent * min(antecedent, consequent))
    
    contingency_table = [[joint, consequent - joint], [antecedent - joint, max(total - antecedent - consequent + joint, 0)]]
    contingency_df = pd.DataFrame(contingency_table, columns=['Consequent', 'Not Consequent'], index=['Antecedent', 'Not Antecedent']).astype(int)
    sns.heatmap(contingency_df, ax=ax1, annot=True, cmap='Blues', square=True, vmin=0, vmax=total, fmt='.0f')
    ax1.set_title('Contingency Table')
    
    v = venn2(subsets=(antecedent - joint, consequent - joint, joint),
              set_labels=['Antecedent', 'Consequent'],
              set_colors=current_palette[:2],
              ax=ax2)
    ax2.set_title("{} Transactions".format(total))

    support_antecedent = antecedent / total
    support_consequent = consequent / total

    support = pd.Series({'Antecedent': support_antecedent,
                         'Consequent': support_consequent})
    support.plot(kind='bar', ax=ax3,
                 color=current_palette[:2], title='Support', ylim=(0, 1), rot=0)
    ax3.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    support_joint = joint / total
    confidence = support_joint / support_antecedent
    lift = support_joint / (support_antecedent * support_consequent)

    _, pvalue = fisher_exact(contingency_table, alternative='greater')

    metrics = pd.Series(
        {'Confidence': confidence, 'Lift': lift, 'p-Value': pvalue})
    metrics.plot(kind='bar', ax=ax4,
                 color=current_palette[2:5], rot=0, ylim=(0, 2))
    ax3.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    for ax, series in {ax3: support, ax4: metrics}.items():
        rects = ax.patches
        labels = ['{:.0%}'.format(x) for x in series.tolist()]
        for rect, label in zip(rects, labels):
            height = min(rect.get_height() + .01, 2.05)
            ax.text(rect.get_x() + rect.get_width() / 2,
                    height, label, ha='center', va='bottom')

    plt.suptitle('Assocation Rule Analysis {Antecedent => Consequent}')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

interact(plot_metrics,
         antecedent=antecedent_widget,
         consequent=consequent_widget,
         joint_percent=joint_widget,
         total=total_widget);

#%%
