from model.lab_color_moment import get_lab_color_moment_features, get_lab_color_moment_feature_names
from model.yuv_color_moment import get_yuv_color_moment_features, get_yuv_color_moment_feature_names

from model.lbp_feature_extraction import get_lbp_features, get_lbp_feature_names
from model.glrlm_feature_extraction import get_glrlm_features, get_glrlm_feature_names
from model.tamura_feature_extraction import get_tamura_features, get_tamura_feature_names

import pandas as pd

def get_cerviscan_features(image_path):
    features = []
    features_name = []
    
    lab_features = get_yuv_color_moment_features(image_path)
    lab_features_name = get_yuv_color_moment_feature_names()
    
    lbp_features = get_lbp_features(image_path)
    lbp_features_name = get_lbp_feature_names()
    
    glrlm_features = get_glrlm_features(image_path)
    glrlm_features_name = get_glrlm_feature_names()
    
    tamura_features = get_tamura_features(image_path)
    tamura_features_name = get_tamura_feature_names()
    
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