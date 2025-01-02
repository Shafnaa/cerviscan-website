import matplotlib.pyplot as plt 
from PIL import Image 
import numpy as np
from itertools import groupby
from model.lbp_feature_extraction import lbp_implementation

class getGrayRumatrix:
    def __init__(self):
        """
        Constructor for the `getGrayRumatrix` class.
        Initializes the object with a `data` attribute to store image data.
        """
        self.data = None
    
    def read_img(self, path=" ", lbp="off"):
        """
        Reads an image from the specified path and converts it to grayscale.

        Parameters:
        - path (str): Path to the image file.
        - lbp (str): Option to use LBP preprocessing. Default is 'off'.

        Returns:
        - np.ndarray: Grayscale image data as a numpy array.
        """
        try:
            if lbp == 'off':
                img = Image.open(path)
                img = img.convert('L')  # Convert to grayscale
                self.data = np.array(img)
            else:
                self.data = lbp_implementation(path)
            return self.data
        except Exception as e:
            print(f"Error reading image: {e}")
            self.data = None
            return None

    def getGrayLevelRumatrix(self, array, theta):
        """
        Computes the Gray-Level Run Length Matrix (GLRLM) for an image.

        Parameters:
        - array (np.ndarray): Grayscale image as a numpy array.
        - theta (list of str): Angles for GLRLM calculation. Supported values are ['deg0', 'deg45', 'deg90', 'deg135'].

        Returns:
        - np.ndarray: GLRLM as a 3D numpy array.
        """
        P = array
        x, y = P.shape
        min_pixels = np.min(P).astype(np.int32)  # Minimum pixel value
        max_pixels = np.max(P).astype(np.int32)  # Maximum pixel value
        run_length = max(x, y)  # Maximum run length in pixels
        num_level = max_pixels - min_pixels + 1
        
        # Define pixel sequences for different angles
        deg0 = [val.tolist() for sublist in np.vsplit(P, x) for val in sublist]
        deg90 = [val.tolist() for sublist in np.split(np.transpose(P), y) for val in sublist]
        diags = [P[::-1, :].diagonal(i) for i in range(-P.shape[0]+1, P.shape[1])]
        deg45 = [n.tolist() for n in diags]
        Pt = np.rot90(P, 3)  # Rotate for 135 degrees
        diags = [Pt[::-1, :].diagonal(i) for i in range(-Pt.shape[0]+1, Pt.shape[1])]
        deg135 = [n.tolist() for n in diags]

        def length(l):
            """Computes the length of an iterable."""
            if hasattr(l, '__len__'):
                return np.size(l)
            else:
                return sum(1 for _ in l)

        glrlm = np.zeros((num_level, run_length, len(theta)))
        
        for angle in theta:
            for splitvec in range(0, len(eval(angle))):
                flattened = eval(angle)[splitvec]
                answer = []
                for key, iter in groupby(flattened):
                    answer.append((key, length(iter)))
                for ansIndex in range(0, len(answer)):
                    glrlm[int(answer[ansIndex][0]-min_pixels), int(answer[ansIndex][1]-1), theta.index(angle)] += 1
        
        return glrlm

    def apply_over_degree(self, function, x1, x2):
        """
        Applies a specified function over the GLRLM across all angles.

        Parameters:
        - function (callable): Function to apply.
        - x1 (np.ndarray): Input matrix.
        - x2 (np.ndarray): Second input for the function.

        Returns:
        - np.ndarray: Resulting matrix after applying the function.
        """
        rows, cols, nums = x1.shape
        result = np.ndarray((rows, cols, nums))
        for i in range(nums):
            result[:, :, i] = function(x1[:, :, i], x2)
        result[result == np.inf] = 0
        result[np.isnan(result)] = 0
        return result 

    def calcuteIJ(self, rlmatrix):
        """
        Calculates indices for gray levels (I) and run lengths (J).

        Parameters:
        - rlmatrix (np.ndarray): GLRLM matrix.

        Returns:
        - tuple: (I, J+1) indices for GLRLM.
        """
        gray_level, run_length, _ = rlmatrix.shape
        I, J = np.ogrid[0:gray_level, 0:run_length]
        return I, J+1

    def calcuteS(self, rlmatrix):
        """
        Calculates the sum of all values in the GLRLM.

        Parameters:
        - rlmatrix (np.ndarray): GLRLM matrix.

        Returns:
        - float: Sum of all elements in the GLRLM.
        """
        return np.apply_over_axes(np.sum, rlmatrix, axes=(0, 1))[0, 0]

    #1.SRE
    def getShortRunEmphasis(self,rlmatrix):
            I, J = self.calcuteIJ(rlmatrix)
            numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, rlmatrix, (J*J)), axes=(0, 1))[0, 0]
            S = self.calcuteS(rlmatrix)
            return numerator / S
    
    #2.LRE
    def getLongRunEmphasis(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.multiply, rlmatrix, (J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
    
    #3.GLN
    def getGrayLevelNonUniformity(self,rlmatrix):
        G = np.apply_over_axes(np.sum, rlmatrix, axes=1)
        numerator = np.apply_over_axes(np.sum, (G*G), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
    
    # 4. RLN
    def getRunLengthNonUniformity(self,rlmatrix):
            R = np.apply_over_axes(np.sum, rlmatrix, axes=0)
            numerator = np.apply_over_axes(np.sum, (R*R), axes=(0, 1))[0, 0]
            S = self.calcuteS(rlmatrix)
            return numerator / S

    # 5. RP
    def getRunPercentage(self,rlmatrix):
            gray_level, run_length,_ = rlmatrix.shape
            num_voxels = gray_level * run_length
            return self.calcuteS(rlmatrix) / num_voxels

    # 6. LGLRE
    def getLowGrayLevelRunEmphasis(self,rlmatrix):
            I, J = self.calcuteIJ(rlmatrix)
            numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, rlmatrix, (I*I)), axes=(0, 1))[0, 0]
            S = self.calcuteS(rlmatrix)
            return numerator / S

    # 7. HGL   
    def getHighGrayLevelRunEmphais(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.multiply, rlmatrix, (I*I)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 8. SRLGLE
    def getShortRunLowGrayLevelEmphasis(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
    
    # 9. SRHGLE
    def getShortRunHighGrayLevelEmphasis(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        temp = self.apply_over_degree(np.multiply, rlmatrix, (I*I))
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 10. LRLGLE
    def getLongRunLow(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        temp = self.apply_over_degree(np.multiply, rlmatrix, (J*J))
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.divide, temp, (J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S

    # 11. LRHGLE
    def getLongRunHighGrayLevelEmphais(self,rlmatrix):
        I, J = self.calcuteIJ(rlmatrix)
        numerator = np.apply_over_axes(np.sum, self.apply_over_degree(np.multiply, rlmatrix, (I*I*J*J)), axes=(0, 1))[0, 0]
        S = self.calcuteS(rlmatrix)
        return numerator / S
