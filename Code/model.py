import pandas as pd
import pickle
import statsmodels.api as sm
import argparse

def prep_data(df):
    df = df.rename(columns = {'SibSp':'family_members','Parch':'parents','Embarked':'port'})
    df = df[['PassengerId','Pclass','Sex','Age','family_members','parents','Fare','Cabin','port','Survived']]
    df2 = pd.get_dummies(df, columns=['Pclass','Sex','port'], drop_first=True)
    df2['Age'] = df2['Age'].fillna(df2['Age'].mean())
    df2 = df2.drop('Cabin', axis = 1)
    df2 = sm.add_constant(df2)
    df3 = df2[['Age', 'family_members', 'parents', 'Fare', 'Pclass_2', 'Pclass_3', 'Sex_male', 'port_Q', 'port_S']]
    return df3

def load_model(model_name):
    with open(model_name, 'rb') as file:
        model = pickle.load(file)
    return model

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Titanic Survival Model')
#     parser.add_argument('--data', type=str, default='titanic.csv', required=True)
#     parser.add_argument('--model', type=str, required=True)
#     args = parser.parse_args()
    
#     data = pd.read_csv(args.data)
#     model = load_model(args.model)
#     score = prep_data(data)
#     preds = model.predict(score)
    
#     pd.DataFrame({'survive_p':preds}).to_csv('predictions.csv', index=False)
#     print(preds)
    
# python score.py --data data/titanic.csv --model models/logreg_model.pkl