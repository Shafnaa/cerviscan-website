import numpy as np
import warnings
from model.GrayRumatrix import getGrayRumatrix

warnings.filterwarnings("ignore")

def get_glrlm_names(features, degs):
    """
    Generate feature names for GLRLM (Gray Level Run Length Matrix).

    Parameters:
        features (list): List of feature names.
        degs (list of lists): List of directional angles (e.g., ['deg0', 'deg45', 'deg90', 'deg135']).

    Returns:
        list: Concatenated feature names with directional angles.
    """
    glrlm_features_name = []
    for deg in degs:
        for feature in features:
            glrlm_features_name.append(f"{feature}_{deg[0]}")
    return glrlm_features_name

def get_glrlm_feature_names():
    """
    Get all feature names for GLRLM.

    Returns:
        list: List of GLRLM feature names including directional angles.
    """
    glrlm_features = ['SRE', 'LRE', 'GLN', 'RLN', 'RP', 'LGLRE', 'HGL', 'SRLGLE', 'SRHGLE', 'LRLGLE', 'LRHGLE']
    glrlm_degs = [['deg0'], ['deg45'], ['deg90'], ['deg135']]
    glrlm_features_name = get_glrlm_names(glrlm_features, glrlm_degs)
    
    return glrlm_features_name

def get_glrlm_features(path, lbp='off'):
    """
    Calculate GLRLM features for an image.

    Parameters:
        path (str): Path to the input image.
        lbp (str, optional): If 'on', apply Local Binary Pattern (LBP) transformation. Defaults to 'off'.

    Returns:
        list: Extracted GLRLM feature values.
    """
    test = getGrayRumatrix()
    test.read_img(path, lbp)

    DEG = [['deg0'], ['deg45'], ['deg90'], ['deg135']]

    glrlm_features_value = []

    for deg in DEG:
        test_data = test.getGrayLevelRumatrix(test.data, deg)
        
        # 1. Short Run Emphasis (SRE)
        SRE = test.getShortRunEmphasis(test_data) 
        SRE = float(np.squeeze(SRE))
        
        # 2. Long Run Emphasis (LRE)
        LRE = test.getLongRunEmphasis(test_data)
        LRE = float(np.squeeze(LRE))
        
        # 3. Gray Level Non-Uniformity (GLN)
        GLN = test.getGrayLevelNonUniformity(test_data)
        GLN = float(np.squeeze(GLN))
        
        # 4. Run Length Non-Uniformity (RLN)
        RLN = test.getRunLengthNonUniformity(test_data)
        RLN = float(np.squeeze(RLN))

        # 5. Run Percentage (RP)
        RP = test.getRunPercentage(test_data)
        RP = float(np.squeeze(RP))
        
        # 6. Low Gray Level Run Emphasis (LGLRE)
        LGLRE = test.getLowGrayLevelRunEmphasis(test_data)
        LGLRE = float(np.squeeze(LGLRE))
        
        # 7. High Gray Level Run Emphasis (HGL)
        HGL = test.getHighGrayLevelRunEmphais(test_data)
        HGL = float(np.squeeze(HGL))
        
        # 8. Short Run Low Gray Level Emphasis (SRLGLE)
        SRLGLE = test.getShortRunLowGrayLevelEmphasis(test_data)
        SRLGLE = float(np.squeeze(SRLGLE))
        
        # 9. Short Run High Gray Level Emphasis (SRHGLE)
        SRHGLE = test.getShortRunHighGrayLevelEmphasis(test_data)
        SRHGLE = float(np.squeeze(SRHGLE))
        
        # 10. Long Run Low Gray Level Emphasis (LRLGLE)
        LRLGLE = test.getLongRunLow(test_data)
        LRLGLE = float(np.squeeze(LRLGLE))
        
        # 11. Long Run High Gray Level Emphasis (LRHGLE)
        LRHGLE = test.getLongRunHighGrayLevelEmphais(test_data)
        LRHGLE = float(np.squeeze(LRHGLE))

        glrlm_features_value_per_deg = [SRE, LRE, GLN, RLN, RP, LGLRE, HGL, SRLGLE, SRHGLE, LRLGLE, LRHGLE]
        
        for value in glrlm_features_value_per_deg:
            glrlm_features_value.append(value)

    return glrlm_features_value 

def get_glrlm_on(path):
    """
    Calculate GLRLM features for an image with LBP transformation.

    Parameters:
        path (str): Path to the input image.

    Returns:
        list: Extracted GLRLM feature values.
    """
    return get_glrlm_features(path, lbp='on')