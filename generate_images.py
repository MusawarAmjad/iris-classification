# generate_images.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Create iris_visualization.png
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
scatter1 = sns.scatterplot(x='sepal length (cm)', y='sepal width (cm)', 
                          hue='species', data=df, s=100, alpha=0.8)
plt.title('Sepal Length vs Sepal Width', fontsize=14, fontweight='bold')
plt.legend(title='Species')

plt.subplot(1, 2, 2)
scatter2 = sns.scatterplot(x='petal length (cm)', y='petal width (cm)', 
                          hue='species', data=df, s=100, alpha=0.8)
plt.title('Petal Length vs Petal Width', fontsize=14, fontweight='bold')
plt.legend(title='Species')

plt.tight_layout()
plt.savefig('static/iris_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

# Create feature_importance.png (sample data)
plt.figure(figsize=(10, 6))
features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
importance = [0.1, 0.2, 0.4, 0.3]  # Sample values - petal features are usually more important

bars = plt.barh(features, importance, color=['#6f42c1', '#6f42c1', '#4e2a8e', '#4e2a8e'])
plt.xlabel('Importance', fontweight='bold')
plt.title('Feature Importance in Iris Classification', fontsize=14, fontweight='bold')
plt.xlim(0, 0.5)

# Add value labels on bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.2f}', ha='left', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('static/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Create confusion_matrix.png (sample data)
plt.figure(figsize=(8, 6))
cm = np.array([[10, 0, 0], [0, 9, 1], [0, 0, 10]])  # Sample confusion matrix

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names,
            cbar=False)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual Species', fontweight='bold')
plt.xlabel('Predicted Species', fontweight='bold')

plt.tight_layout()
plt.savefig('static/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("All visualization images created successfully!")
print("Files saved in static/ folder:")
print("- iris_visualization.png")
print("- feature_importance.png") 
print("- confusion_matrix.png")