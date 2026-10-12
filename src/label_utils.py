import numpy as np
import pandas as pd
TEETH=list(range(2,16))+list(range(18,32))
def clean(df):
    for c in df.select_dtypes(include='number'):
        df.loc[(df[c].abs()>0)&(df[c].abs()<1e-70),c]=np.nan
    return df

def label(perio, standalone=True):
    def matrix(prefix):
        a=np.stack([perio[[f'OHX{t:02d}{prefix}{s}' for s in 'DSPA']].to_numpy(float) for t in TEETH],axis=1)
        a[(a==99)|(a>90)]=np.nan
        return a
    al,pd_=matrix('LA'),matrix('PC')
    def n(a,cut): return (a>=cut).any(axis=2).sum(axis=1)
    p5=n(pd_,5)>0
    severe=(n(al,6)>=2)&p5
    moderate=(n(al,4)>=2)|(n(pd_,5)>=2)
    # Mild AL condition counts interproximal sites; PD >=4 needs distinct teeth.
    mild=(((al>=3).sum(axis=(1,2))>=2)&(n(pd_,4)>=2))|p5 if standalone else (n(al,3)>=2)&((n(pd_,4)>=2)|p5)
    result=np.select([severe,moderate,mild],['Severe','Moderate','Mild'],default='No Periodontitis').astype(object)
    result[~(np.isfinite(al).any(axis=(1,2))|np.isfinite(pd_).any(axis=(1,2)))]=None
    return pd.Series(result,index=perio.index)
