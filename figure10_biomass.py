from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
root=Path(__file__).resolve().parent
source = root / "final_finetuned_master_dataset.csv"
df=pd.read_csv(source)
x=df['Raw_Final_Leaf_Area_cm2'].to_numpy(float)
y=df['Measured_Final_Dry_Weight_g'].to_numpy(float)
fit=linregress(x,y)
pred=fit.intercept+fit.slope*x
result=df[['Plant_ID','Raw_Final_Leaf_Area_cm2','Measured_Final_Dry_Weight_g']].copy()
result['Fitted_Biomass_g']=pred
result.to_csv(root/'figure10_corrected_values.csv',index=False)
fig,ax=plt.subplots(figsize=(17/2.54,10/2.54))
ax.scatter(y,pred,s=34)
for label,actual,fitted in zip(df.Plant_ID,y,pred):
    ax.annotate(label,(actual,fitted),xytext=(3,3),textcoords='offset points',fontsize=6.5)
lo=min(0,y.min(),pred.min());hi=max(y.max(),pred.max())*1.08
ax.plot([lo,hi],[lo,hi],'--',color='grey',lw=1,label='1:1 line')
ax.set(xlabel='Measured final dry biomass (g)',ylabel='Fitted final dry biomass (g)',title=f'Raw HSV leaf area regression (in-sample)\nR² = {fit.rvalue**2:.3f}; RMSE = {np.mean((y-pred)**2)**.5:.3f} g; n = {len(y)}')
ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
fig.savefig(root/'figure10_corrected_raw_HSV.png',dpi=600)
plt.show()
print('n:',len(y),'slope:',fit.slope,'intercept:',fit.intercept,'R2:',fit.rvalue**2,'RMSE:',np.mean((y-pred)**2)**.5)
