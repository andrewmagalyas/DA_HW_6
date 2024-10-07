import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

penguins = sns.load_dataset('penguins')
penguins.dropna(inplace=True)
penguins = pd.get_dummies(penguins, columns=['island', 'sex'])

X = penguins.drop('species', axis=1)
y = penguins['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f'Cross-validation scores: {scores}')

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f'Test accuracy: {accuracy}')

param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f'Best parameters: {grid_search.best_params_}')

y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
