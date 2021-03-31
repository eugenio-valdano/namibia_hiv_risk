# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
from matplotlib.colors import hsv_to_rgb , rgb_to_hsv, LinearSegmentedColormap #,LogNorm, LinearSegmentedColormap, ListedColormap, BoundaryNorm
import pickle
#import igraph as ig
#import louvain
import networkx as nx
from pandas import DataFrame, read_csv, isnull, merge, concat
import sys
PYTHON3 = sys.version_info[0] >= 3

# load HTML engine, if it exists
try:
    if __IPYTHON__:
        from IPython.core.display import HTML
    else:
        HTML = lambda x : x
except NameError:
    HTML = lambda x : x



#########
## COLORS

class myColor(object):

    @staticmethod
    def rgb_to_cmyk(r,g,b, cmyk_scale=100., rgb_scale=255.):

        if np.abs(r)<1e-6 and np.abs(g)<1e-6 and np.abs(b)<1e-6:
            # black
            k, c, m, y = 0., 1., 1., 1.

        else:
            k = 0. #1. - max([r,g,b])
            c = (1.-r-k)/(1.-k)
            m = (1.-g-k)/(1.-k)
            y = (1.-b-k)/(1.-k)

        return np.array([c*cmyk_scale, m*cmyk_scale, y*cmyk_scale, k*cmyk_scale])

        """
        if np.abs(r)<1e-6 and np.abs(g)<1e-6 and np.abs(b)<1e-6:
            # black
            return np.array([0., 0., 0., cmyk_scale])



        # rgb [0,255] -> cmy [0,1]
        c = 1 - r / float(rgb_scale)
        m = 1 - g / float(rgb_scale)
        y = 1 - b / float(rgb_scale)

        # extract out k [0,1]
        min_cmy = min(c, m, y)
        c = (c - min_cmy)
        m = (m - min_cmy)
        y = (y - min_cmy)
        k = min_cmy

        # rescale to the range [0,cmyk_scale]
        return np.array([c*cmyk_scale, m*cmyk_scale, y*cmyk_scale, k*cmyk_scale])
        """

    @staticmethod
    def cmyk_to_rgb(c,m,y,k, cmyk_scale=100., rgb_scale=255.):
        r = rgb_scale * (1.-c/cmyk_scale) * (1.-k/cmyk_scale)#rgb_scale*(1.0-(c+k)/float(cmyk_scale))
        g = rgb_scale * (1.-m/cmyk_scale) * (1.-k/cmyk_scale)#rgb_scale*(1.0-(m+k)/float(cmyk_scale))
        b = rgb_scale * (1.-y/cmyk_scale) * (1.-k/cmyk_scale)#rgb_scale*(1.0-(y+k)/float(cmyk_scale))
        return np.array([r,g,b])

    # CONSTRUCTOR
    def __init__(self,x,rgb=True,norm=1.):

        if type(x) == str:
            # hex
            self._hex = x
        elif len(x) == 4:
            # cmyk
            if rgb:
                self._rgb = np.array(x[:3])/float(norm)
            else:
                self.cmyk = np.array(x)/float(norm)
        else:
            if rgb:
                self._rgb = np.array(x)/float(norm)
                #print ('alap', x, self._rgb)
            else:
                self._hsv = np.array(x)/float(norm)

    def hex_to_rgb(self,value):
        value = value.lstrip('#')
        lv = len(value)
        return np.array([ int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3) ], dtype=float)/255.

    def rgb_to_hex(self,value):
        #return '#%02x%02x%02x' % tuple(value*255.)
        v = np.round( 255*np.array(value) )
        v = map(int, v)
        return "#{0:02x}{1:02x}{2:02x}".format(*v)


    def rgb_to_hsv(self,value):
        return rgb_to_hsv(value)

    def hsv_to_rgb(self,value):
        return hsv_to_rgb(value)

    def hex_to_hsv(self,value):
        rgb = self.hex_to_rgb(value)
        return self.rgb_to_hsv(rgb)

    def hsv_to_hex(self,value):
        rgb = self.hsv_to_rgb(value)
        return self.rgb_to_hex(rgb)


    # RGB
    @property
    def rgb(self):
        if not hasattr(self,'_rgb'):

            if hasattr(self,'_hex'):
                self._rgb = self.hex_to_rgb(self._hex)
            else:
                self._rgb = self.hsv_to_rgb(self._hsv)

        return self._rgb

    @rgb.setter
    def rgb(self,x):
        assert all(x<=1.) and all(x>=0.), 'must be between 0 and 1.'
        self._rgb = x
        del self.hex
        del self.hsv

    @rgb.deleter
    def rgb(self):
        if hasattr(self,'_rgb'):
            del self._rgb

    # HSV
    @property
    def hsv(self):

        if not hasattr(self,'_hsv'):

            if hasattr(self,'_hex'):
                self._hsv = self.hex_to_hsv(self._hex)
            else:
                self._hsv = self.rgb_to_hsv(self._rgb)

        return self._hsv

    @hsv.setter
    def hsv(self,x):
        assert all(np.abs(x)<=1.), 'no greater than 1.'
        self._hsv = x
        del self.rgb
        del self.hex

    @hsv.deleter
    def hsv(self):
        if hasattr(self,'_hsv'):
            del self._hsv

    # HEX
    @property
    def hex(self):
        if not hasattr(self,'_hex'):
            if hasattr(self,'_rgb'):
                self._hex = self.rgb_to_hex(self._rgb)
            else:
                self._hex = self.hsv_to_hex(self._hsv)

        return self._hex

    @hex.setter
    def hex(self,x):
        assert type(x)==str and x[0]=='#' and len(x)==7, 'Wrong format.'
        self._hex = x
        del self.rgb
        del self.hsv

    @hex.deleter
    def hex(self):
        if hasattr(self,'_hex'):
            del self._hex

    # CMYK
    @property
    def cmyk(self):
        if not hasattr(self, '_cmyk'):
            my_rgb = self.rgb
            self._cmyk = self.rgb_to_cmyk(my_rgb[0], my_rgb[1], my_rgb[2], rgb_scale=1.0, cmyk_scale=1.0)
        return self._cmyk

    @cmyk.setter
    def cmyk(self, x):
        assert np.all(x>=0.) and np.all(x<=1.), 'wrong format: must be between 0 and 1'
        self._cmyk = np.array(x)
        self._rgb = self.cmyk_to_rgb(x[0], x[1], x[2], x[3], rgb_scale=1.0, cmyk_scale=1.0)
        del self.hsv

    @cmyk.deleter
    def cmyk(self):
        if hasattr(self,'_cmyk'):
            del self._cmyk


    # LUMINANCE
    @property
    def L (self):
        if not hasattr(self,'_L'):
            e = []
            for x in self.rgb:
                if x <= 0.03928:
                    y = x/12.92
                else:
                    y = np.power( (x+0.055)/1.055, 2.4 )
                e.append(y)

            self._L = 0.2126*e[0] + 0.7152*e[1] + 0.0722*e[2]

        return self._L

    @L.setter
    def L(self,x):
        assert False, 'You cannot set luminance.'

    @L.deleter
    def L(self):
        if hasattr(self,'_L'):
            del self._L


    # COMPLEMENTARY
    def complementary (self):
        x = self.hsv.copy()
        x[0] = (x[0]+0.5)%1.
        return myColor(x, rgb=False)


    def __str__(self):
        return 'COLOR {}'.format(self.hex)

    def __repr__(self):
        return self.__str__()

    def display(self):
        s = '<span style="color:{hex};font-weight:bold">COLOR {hex}</span>'.format(hex=self.hex)
        return HTML(s)

def display_colors(lc, labels=None):
    v = []
    for x in lc:
        if type(x)==str:
            v.append(myColor(x))
        else:
            v.append(x)
    s = ''

    if labels is None:
        for x in v:
            s += '<span style="color:{hex};font-weight:bold">COLOR {hex}</span><br>'.format(hex=x.hex)
    else:
        for x, name in zip(v, labels):
            s += '<span style="color:{hex};font-weight:bold">{label}</span><br>'.format(hex=x.hex, label=name)

    return HTML(s)


def linear_color_interpolator(ci, cf, n, rgb=False, endpoint=True): # in array di rgb, normalizzati a 1

        #ek = []
        for h in np.linspace(0, 1, n, endpoint=endpoint):
            if rgb:
                colore = ci.rgb + (cf.rgb-ci.rgb)*h
            else:
                colore = ci.hsv + (cf.hsv-ci.hsv)*h
            #ek.append(myColor(colore,rgb=True))
            yield myColor(colore,rgb=rgb)


def display_colors(lc, labels=None):
    v = []
    for x in lc:
        if type(x)==str:
            v.append(myColor(x))
        else:
            v.append(x)
    s = ''

    if labels is None:
        for x in v:
            s += '<span style="color:{hex};font-weight:bold">COLOR {hex}</span><br>'.format(hex=x.hex)
    else:
        for x, name in zip(v, labels):
            s += '<span style="color:{hex};font-weight:bold">{label}</span><br>'.format(hex=x.hex, label=name)

    return HTML(s)



###########
# WISE MULTIPLIER
def wise_multiplier(M, left=None, right=None):

    """
    Y = wise_multiplier(M, left=v, right=w) means Y_{ij} = v_i M_{ij} w_j (no einstein sum).
    If left is None, then  Y_{ij} = M_{ij} w_j.
    If right is None, then  Y_{ij} = v_i M_{ij}.
    """

    ek = M.copy()
    if left is not None:
        ek = np.transpose(np.transpose(ek)*left)
    if right is not None:
        ek = ek*right
    return ek




def countcol(df, x2, ascending=False, colname='COUNT'):
    if type(x2) == list or type(x2) == tuple:
        x = x2
    else:
        x = [x2]
    ek = df[x].groupby(x2).size().reset_index().sort_values(by=0, ascending=ascending).reset_index(drop=True).rename(columns={0: colname})
    ek = ek.assign(**{colname+'_FRAC': ek[colname]/ek[colname].sum()})
    ek = ek.assign(**{colname + '_FRACCUM': ek[colname+'_FRAC'].cumsum()})
    return ek
