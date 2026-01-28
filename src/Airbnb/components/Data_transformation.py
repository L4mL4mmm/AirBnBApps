import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.Airbnb.exception import customexception
from src.Airbnb.logger import logging

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler

from src.Airbnb.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('Artifacts','Preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    
    def get_data_transformation(self):
        
        try:
            logging.info('Data Transformation initiated')


            numerical_cols   = ['amenities','accommodates','bathrooms','latitude','longitude','host_response_rate','number_of_reviews','review_scores_rating','bedrooms','beds']
            categorical_cols = ['property_type','room_type','bed_type','cancellation_policy','cleaning_fee','city','host_identity_verified','instant_bookable','host_has_profile_pic']

            property_type_cat = sorted(list(set(['Apartment', 'House', 'Condominium', 'Townhouse', 'Loft', 'Other', 'Guesthouse', 'Bed & Breakfast', 'Bungalow', 'Villa', 'Dorm', 'Guest suite', 'Camper/RV', 'Timeshare', 'Cabin', 'In-law', 'Hostel', 'Boutique hotel', 'Boat', 'Serviced apartment', 'Tent', 'Castle', 'Vacation home', 'Yurt', 'Hut', 'Treehouse', 'Chalet', 'Earth House', 'Tipi', 'Train', 'Cave', 'Casa particular', 'Parking Space', 'Lighthouse', 'Island', 'Entire condo', 'Entire home', 'Entire rental unit', 'Entire guest suite', 'Entire vacation home', 'Private room in condo', 'Private room in rental unit', 'Private room in home', 'Room in hotel', 'Room in boutique hotel', 'Room in aparthotel', 'Entire townhouse', 'Entire loft', 'Private room in townhouse', 'Private room in loft', 'Shared room in rental unit', 'Shared room in home', 'Shared room in condo', 'Tiny home', 'Entire cottage', 'Private room in cottage', 'Entire guesthouse', 'Private room in guest suite', 'Private room in guesthouse', 'Entire bungalow', 'Private room in bungalow', 'Entire villa', 'Private room in villa', 'Entire place', 'Private room', 'Shared room', 'Hotel room', 'Entire serviced apartment', 'Private room in serviced apartment', 'Shared room in hostel', 'Private room in hostel', 'Room in bed and breakfast', 'Barn', 'Bus', 'Campsite', 'Dome', 'Dome house', 'Farm stay', 'Houseboat', 'Kezhan', 'Minsu', 'Religious building', 'Riad', 'Shepherd\'s hut', 'Shipping container', 'Tower', 'Trullo', 'Windmill'])))
            
            room_type_cat = sorted(list(set(['Entire home/apt', 'Private room', 'Shared room', 'Hotel room', 'Entire home', 'Entire condo', 'Entire guest suite', 'Private room in rental unit', 'Private room in home', 'Entire rental unit'])))
            
            bed_type_cat = ['Real Bed', 'Futon', 'Pull-out Sofa', 'Airbed', 'Couch']
            cancellation_policy_cat = ['strict', 'moderate', 'flexible', 'super_strict_30', 'super_strict_60']
            cleaning_fee_cat = ['True', 'False', '0', '1']
            
            city_cat = ['NYC', 'SF', 'DC', 'LA', 'Chicago', 'Boston']
            host_has_profile_pic_cat = ['t', 'f']
            host_identity_verified_cat = ['t', 'f']
            instant_bookable_cat = ['t', 'f', '0', '1']

            logging.info('Pipeline Initiated')
            
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())])
            
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[property_type_cat, room_type_cat, bed_type_cat, cancellation_policy_cat, cleaning_fee_cat, city_cat, host_has_profile_pic_cat, host_identity_verified_cat, instant_bookable_cat], handle_unknown='use_encoded_value', unknown_value=-1)),
                ('scaler',StandardScaler())])
            
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])
            
            return preprocessor
        
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise customexception(e,sys)
            
    
    def initialize_data_transformation(self,train_path,val_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            val_df=pd.read_csv(val_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("read train, val and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Val Dataframe Head : \n{val_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')
            
            preprocessing_obj = self.get_data_transformation()

            for df in [train_df, val_df, test_df]:
                if df['host_response_rate'].dtype == 'object':
                     df['host_response_rate'] = df['host_response_rate'].astype(str).str.replace('%', '').replace('nan', np.nan).replace('N/A', np.nan).astype(float)
                
                # Robust handling for missing cleaning_fee
                if 'cleaning_fee' not in df.columns:
                    logging.info("cleaning_fee missing, filling with '0'")
                    df['cleaning_fee'] = '0'
                
                if 'bed_type' not in df.columns:
                    df['bed_type'] = 'Real Bed'
                
                if 'cancellation_policy' not in df.columns:
                    df['cancellation_policy'] = 'flexible'

                df['cleaning_fee'] = df['cleaning_fee'].astype(str)
                
                df['amenities'] = [len(str(amenity).split(',')) for amenity in df['amenities']]


            logging.info("Host Response Rate converted and amenities fixed")

            target_column_name = 'log_price'
            drop_columns = [target_column_name,'id',"name","description","first_review","host_since","last_review","neighbourhood","thumbnail_url", "zipcode"]
            
            price_99_percentile = train_df['log_price'].quantile(0.99)
            logging.info(f"Capping log_price at 99th percentile: {price_99_percentile:.4f}")
            
            train_df = train_df[train_df['log_price'] <= price_99_percentile]

            # Safely drop columns (ignore if not found)
            existing_drop_cols = [col for col in drop_columns if col in train_df.columns]
            input_feature_train_df = train_df.drop(columns=existing_drop_cols, axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            existing_drop_cols_val = [col for col in drop_columns if col in val_df.columns]
            input_feature_val_df = val_df.drop(columns=existing_drop_cols_val, axis=1)
            target_feature_val_df=val_df[target_column_name]

            existing_drop_cols_test = [col for col in drop_columns if col in test_df.columns]
            input_feature_test_df=test_df.drop(columns=existing_drop_cols_test, axis=1)
            target_feature_test_df=test_df[target_column_name]


            logging.info(f'Input Feature Train Dataframe Head : \n{input_feature_train_df.head().to_string()}')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_val_arr = preprocessing_obj.transform(input_feature_val_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            
            logging.info("Applying preprocessing object on training, validation and testing datasets.")

            train_arr = np.concatenate([input_feature_train_arr, np.array(target_feature_train_df).reshape(-1, 1)], axis=1)
            val_arr = np.concatenate([input_feature_val_arr, np.array(target_feature_val_df).reshape(-1, 1)], axis=1)
            test_arr = np.concatenate([input_feature_test_arr, np.array(target_feature_test_df).reshape(-1, 1)], axis=1)


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            logging.info("preprocessing pickle file saved")
            
            return (
                train_arr,
                val_arr,
                test_arr
            )
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise customexception(e,sys)