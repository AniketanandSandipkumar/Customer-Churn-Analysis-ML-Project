import matplotlib.pyplot as plt
import pickle

def plot_feature_importance():
    model = pickle.load(open("models/churn_model_latest.pkl","rb"))

    clf = model.named_steps['classifier']
    importance = clf.feature_importances_

    plt.figure()
    plt.bar(range(len(importance)), importance)
    plt.title("Feature Importance")
    plt.savefig("visualizations/feature_importance.png")

if __name__ == "__main__":
    plot_feature_importance()