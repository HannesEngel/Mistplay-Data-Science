import sys
import json
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle


int_vars = ["x{}".format(var) for var in [10, 14, 18, 19, 20, 21, 22, 6, 7, 9]]
float_vars = ["x{}".format(var) for var in [11,12,13]]
obj_vars = ["x{}".format(var) for var in [1,15,16, 17, 2, 23, 24, 25, 26, 3, 4, 5, 8]]

model = pickle.load(open('finalized_model.sav', 'rb'))
dummy_cols = pickle.load(open('dummy_cols.sav', 'rb'))

# Data cleaning function
def clean_data(input_df):
    out = input_df.copy()
   
    out.x26.fillna('Bidalgo Facebook', inplace=True)
   
    out['x2_list'] = out['x2'].astype(str).str.split('.')
    out['Version'] = out['x2_list'].apply(lambda x : x[0])
    out['Release_1'] = out['x2_list'].apply(lambda x : x[1] if len(x) > 1 else None)
    out['Release_2'] = out['x2_list'].apply(lambda x : x[2] if len(x) > 2 else None)
   
    out['x3'] = out['x3'].astype(str).str.replace("-", "_")
    out['x3_list'] = out['x3'].astype(str).str.split('_')
    out['Make'] = out['x3_list'].apply(lambda x : x[0])
    out['Model'] = out['x3_list'].apply(lambda x : x[1] + "_" + x[2] if len(x) > 2 else None)
      
    cols_to_convert = ['x5', 'x8', 'x16', 'Version']
    for col in cols_to_convert:
        out[col] = out[col].astype(int)
   
    out.drop(['x1', 'x2', 'x2_list', 'x3_list', 'x3', 'x15', 'x17', 'x24', 'x25'], axis=1, inplace=True)
   
    return out

def main():

    try:
        args = sys.argv[1]
        data = json.loads(args)
        df = pd.DataFrame(data, index=[0])

        for var in int_vars:
            df[var] = df[var].astype(int)

        for var in float_vars:
            df[var] = df[var].astype(float)

        for var in obj_vars:
            df[var] = df[var].astype(str)

        clean_df = clean_data(df)

        model_string_cols = ['x4', 'x23', 'x26', 'Release_1', 'Release_2', 'Make', 'Model']
        model_numeric_cols = ['x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13',
                              'x14', 'x16', 'x18', 'x19', 'x20', 'x21', 'x22', 'Version']
        
        df_string = pd.get_dummies(clean_df[model_string_cols].copy())
        df_numeric = clean_df[model_numeric_cols].copy()

        dummy_cols_to_fill = [col for col in dummy_cols if
                              col not in df_string.columns]

        for col in dummy_cols_to_fill:
            df_string[col] = 0

        df_string = df_string[dummy_cols]
        
        modelling_data = pd.concat([df_numeric, df_string], axis=1)

        prediction = model.predict(modelling_data)
        
        results = {
            'prediction': prediction[0]
        }

        print(str(results))
        sys.stdout.flush()

    except Exception as e:
        results = {'error': str(e)}
        print(str(results))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
