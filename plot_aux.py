import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances
import matplotlib.cm as cm
from scipy.cluster.hierarchy import  dendrogram 
from sklearn.preprocessing import MinMaxScaler,  LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, classification_report, precision_recall_curve,cohen_kappa_score
 
        
def bar_plot(column:str, df:pd.DataFrame):

    hero_counts = df.groupby(f'{column}')['name'].count().reset_index(name='hero_count')

    plt.figure(figsize=(16,6))
    bars = plt.bar(hero_counts[f'{column}'], hero_counts['hero_count'], color='#008000')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom')
    plt.xlabel(f'{column}')
    plt.ylabel("Quantidade de Heróis")
    plt.title(f"Quantidade de Heróis por {column}")
    plt.xticks(rotation=90) 
    plt.show()

def cramers_V(var1,var2) :
    crosstab = np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building
    stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
    obs = np.sum(crosstab) # Number of observations
    mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
    return (stat/(obs*mini))

def matrix_correlation_plot(df_features):
    rows=[]
    for var1 in df_features.columns.tolist():
        col=[]
        for var2 in df_features.columns.tolist():
            cramers = cramers_V(df_features[var1],df_features[var2])
            col.append(round(cramers,2))
        rows.append(col)
    cramers_results = np.array(rows) 
    corr_matrix = pd.DataFrame(cramers_results,columns = df_features.columns, index=df_features.columns)
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.title(f"Cramer correlation")
    sns.heatmap(corr_matrix, annot=True, cmap="YlGn") #cmap="RdPu"
    return corr_matrix

def plot_boxplot(df:pd.DataFrame, numeric_features:List[str], hue:str):
    fig, axes = plt.subplots(4, 3, figsize=(15, 5 * 3))

    axes = axes.flatten()

    for i, coluna in enumerate(numeric_features):
        sns.boxplot(hue=f'{hue}', y=coluna, data=df, ax=axes[i], palette="Set2")
        axes[i].set_title(f'Distribuição do {coluna} por {hue}', fontsize=12)
        # axes[i].set_xlabel('Cluster', fontsize=10)
        axes[i].set_ylabel(coluna, fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def calculate_anova(df:pd.DataFrame, numeric_var:str, categorical_vars:List[str]) -> pd.DataFrame:
    anova_results = []
    for var in categorical_vars:
        model = ols(f'{numeric_var} ~ C({var})', data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_table['variable'] = var
        anova_results.append(anova_table.reset_index())

    anova_df = pd.concat(anova_results, ignore_index=True)
    return anova_df


def plot_bar_list_features(df:pd.DataFrame, feature_list:List[str]):
    fig, axes = plt.subplots(4, 2, figsize=(15, 5 * 3))
    fig.suptitle('TOP 10 : Quantidade de heróis por features', fontsize=14)

    axes = axes.flatten()
    for i, column in enumerate(feature_list):

        hero_counts = df.groupby(f'{column}').count()['name'].sort_values(ascending=False).reset_index(name='hero_count')[0:10]
        hero_counts[f'{column}'] = hero_counts[f'{column}'].astype(str)
        # colors = hero_counts[f'{column}'].map({'False': 'lightgreen', 'True': 'green'})
        bars = axes[i].bar(hero_counts[f'{column}'], hero_counts['hero_count'], color='green')
        for bar in bars:
            yval = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2, yval + 1, int(yval), ha='center', va='bottom')

        axes[i].set_title(f'{column}', fontsize=12)
        # axes[i].set_xlabel('Cluster', fontsize=10)
        # axes[i].set_ylabel(column, fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylim(0, max(hero_counts['hero_count']) + 100)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    

def plot_hist(df:pd.DataFrame, numeric_features:List[str]):

    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(numeric_features, 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[feature], kde=True, bins=20, color='royalblue')
        plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.show()

def plot_silhouette(dist_matrix, data_to_cluster, n_clusters=2):
    cluster_labels = data_to_cluster['Hierarchical_Cluster']
    silhouette_avg = silhouette_score(dist_matrix, cluster_labels)

    y_lower = 10
    sample_silhouette_values = silhouette_samples(dist_matrix, cluster_labels)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_xlim([-0.1, 1])
    # ax1.set_ylim([0, 1000])
    ax1.set_ylim([0, len(dist_matrix) + (n_clusters + 1) * 10])
    
    for i in [1, 2]:
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.set_title("The silhouette plot for the various clusters")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(dist_matrix[:, 0], dist_matrix[:, 1], marker='.', s=100, lw=0, alpha=0.9,
                c=colors, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for Hierarchical clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    plt.show()

def plot_dendograma(linkage_matrix, labels):
    plt.figure(figsize=(20, 10))
    dendrogram(linkage_matrix, labels=labels)
    plt.title('Dendrograma - Clusterização Hierárquica')
    plt.show()
    
def plot_confusion_matrix(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.show() 
    
def plot_predict_true_values_scatter(y_test,y_pred ):
    plt.figure(figsize=(8, 6))

    plt.scatter(range(len(y_test)), y_test, alpha=0.7, color='green', label='True Values')

    plt.scatter(range(len(y_pred)), y_pred, alpha=0.7, color='blue', label='Predicted Values')

    plt.title("Predicted vs. True Weight (Comparison)")
    plt.xlabel("Index")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True)
    plt.show()
  