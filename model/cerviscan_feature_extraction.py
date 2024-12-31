from model.lab_color_moment import get_lab_color_moment, get_feature_name
from model.lbp_extraction import get_lbp_features, get_lbp_name
from model.glrlm_extraction import get_glrlm, get_glrlm_name
from model.tamura_features import get_tamura, get_tamura_name
import pandas as pd

def get_cerviscan_features(image_path):
    features = []
    features_name = []
    
    lab_features = get_lab_color_moment(image_path)
    lab_features_name = get_feature_name()
    
    lbp_features = get_lbp_features(image_path)
    lbp_features_name = get_lbp_name()
    
    glrlm_features = get_glrlm(image_path)
    glrlm_features_name = get_glrlm_name()
    
    tamura_features = get_tamura(image_path)
    tamura_features_name = get_tamura_name()
    
    features.extend(lab_features)
    features.extend(lbp_features)
    features.extend(glrlm_features)
    features.extend(tamura_features)
    
    features_name.extend(lab_features_name)
    features_name.extend(lbp_features_name)
    features_name.extend(glrlm_features_name)   
    features_name.extend(tamura_features_name)
    
    df_features = pd.DataFrame([features], columns=features_name)
    df_features = df_features.loc[:, (df_features != 1).any()]
    
    return df_features