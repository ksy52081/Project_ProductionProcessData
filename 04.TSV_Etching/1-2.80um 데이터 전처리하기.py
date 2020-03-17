
##### TSV_80um #######

##### 필요한 라이브러리 호출하기 #####
import pandas as pd 
import numpy as np

##### csv 파일들의 이름 리스트 #####
file80_list = ['FDC_TSV80_ (14).CSV', 'FDC_TSV80_ (15).CSV', 'FDC_TSV80_ (16).CSV', 'FDC_TSV80_ (17).CSV', 'FDC_TSV80_ (18).CSV', 'FDC_TSV80_ (19).CSV', 'FDC_TSV80_ (20).CSV', 'FDC_TSV80_ (21).CSV', 'FDC_TSV80_ (22).CSV', 'FDC_TSV80_ (23).CSV', 'FDC_TSV80_ (24).CSV', 'FDC_TSV80_ (25).CSV', 'FDC_80um_Fault_C4F8_90to85_SCCM.CSV','FDC_80um_Fault_O2_13to11_SCCM.CSV', 'FDC_80um_Fault_PlatenPower_17to15_Watt.CSV.CSV', 'FDC_80um_Fault_SF6_130to122_SCCM.CSV', 'FDC_80um_Fault_SourcePower_800to760_Watt.CSV']

#### 변수 리스트 ####
var_list = ['14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '_C4F8_90to85_SCCM', '_O2_13to11_SCCM', '_PlatenPower_17to15_Watt', '_SF6_130to122_SCCM', '_SourcePower_800to760_Watt']

#### 1차 Preprocessing 데이터를 가져오고 불필요한 것들을 쳐내자. ####
t=0
for i in file80_list:
    file_path = str('F:/JupyterLab/AdvancedIntelligence19/AI_Homework02/TSV ETCH/[FDC] TSV/[FDC] TSV_80um/%s'%i)
    
    ##### DataFrame 변수를 만들고 CSV파일의 데이터를 DF형태로 가져온다.#####
    globals()['df_{}'.format(var_list[t])]=pd.read_csv(file_path)
    ##### 앞부분 필요 없는 데이터는 쳐낸다. #####
    globals()['df_{}'.format(var_list[t])] = globals()['df_{}'.format(var_list[t])].drop([0,1,2,3,4,6])
    ##### 인덱스가 깔끔하지 않으니 다시 인덱스를 리셋하자 #####
    globals()['df_{}'.format(var_list[t])] = globals()['df_{}'.format(var_list[t])].reset_index(drop=True)

    #### Time열을 인덱스로 삼고 싶다 -  데이터프레임의 칼럼 이름을 0번째 인덱스 열의 값으로 수정한다. ####
    globals()['df_{}'.format(var_list[t])].columns =globals()['df_{}'.format(var_list[t])].iloc[0]
    #### Time열을 인덱스로 삼고 싶다 -  0번째 인덱스 열의 값은 지운다.
    globals()['df_{}'.format(var_list[t])] = globals()['df_{}'.format(var_list[t])].reindex(globals()['df_{}'.format(var_list[t])].index.drop(0))
    globals()['timetable_{}'.format(var_list[t])] = list(globals()['df_{}'.format(var_list[t])]["Time"])
    t=t+1

#### 숫자가 아닌 데이터를 숫자로 표현하고, NAN 값도 변형시키자. ####
####Process Phase 의 gasstab : 0 Process:1, pumpout:2 ####
for i in var_list:
    globals()['df_{}'.format(i)].loc[globals()['df_{}'.format(i)]["Process Phase"]=="GasStab","Process Phase"]= 0
    globals()['df_{}'.format(i)].loc[globals()['df_{}'.format(i)]["Process Phase"]=="Process","Process Phase"]= 1
    globals()['df_{}'.format(i)].loc[globals()['df_{}'.format(i)]["Process Phase"]=="Pumpout","Process Phase"]= 2
    
    
#### ASE Phase : null값은 -9999 pass:0 Etch:1 ####
    globals()['df_{}'.format(i)].loc[globals()['df_{}'.format(i)]["ASE Phase"]=="Pass","ASE Phase"]= 0
    globals()['df_{}'.format(i)].loc[globals()['df_{}'.format(i)]["ASE Phase"]=="Etch","ASE Phase"]= 1
    globals()['df_{}'.format(i)].loc[globals()['df_{}'.format(i)]["ASE Phase"]=='    ',"ASE Phase"]= -9999
    
    
    ###globals()['df2_{}'.format(i)]= globals()['df_{}'.format(i)].fillna({"ASE Phase":-1})
    ##'    '
### ASE Cycles, ASEDounCount도 빈칸은 -9999 ####
    globals()['df_{}'.format(i)].loc[globals()['df_{}'.format(i)]["ASE Cycles"]=='    ',"ASE Cycles"]= -9999
    globals()['df_{}'.format(i)].loc[globals()['df_{}'.format(i)]["ASE Downcount"]=='    ',"ASE Downcount"]= -9999

###데이터 칼럼의 데이터들의 속성을 float로 변경하기 ####
    globals()['df_{}'.format(i)]= globals()['df_{}'.format(i)].astype({'[1] C4F8':'float', '[2] SF6':'float', '[3] O2':'float', 'Pressure':'float', 'RF1 Power':'float', 'RF1 Reflected':'float', 'RF Peak':'float', 'RF Bias':'float', 'RF1 Load':'float', 'RF1 Tune':'float', 'RF2 Power':'float', 'RF2 Reflected':'float', 'RF2 Load':'float', 'RF2 Tune':'float', 'APC Angle':'float', 'He Leakup Rate':'float', 'He Pressure':'float', 'He Flow':'float', 'Process Phase':'float', 'ASE Phase':'float', 'ASE Cycles':'float', 'ASE Downcount':'float'})


#### 에러와 정상 샘플 17개에서 모두 공통적으로으로 가지는 측정 시간축 찾기 ####

overlapped_time_list = timetable_14
for i in var_list:
    temp = set(overlapped_time_list)
    temp_new = set(globals()['timetable_{}'.format(i)])
    overlapped_time_list = list(temp & temp_new)

#### 혹시 모르는 겹치는 것 제거하고, str으로 바꿔주자. ####
overlapped_time_list = [int (i) for i in overlapped_time_list]
overlapped_time_list.sort()
overlapped_time_list_str = [str (i) for i in overlapped_time_list]

#### 데이터프레임의 열 이름 출력
column_Name_list = list(df_14)
test = column_Name_list[1:]

#### 혹시 같은 시간대에 여러 값들이 출력된다면, group by로 평균값으로 바꾸자.
##### 그룹화 함수  ##### time 에 따라 group by 하니 자연스럽게 time 이 인덱스가 되었다.
for i in var_list:
    #globals()['df_{}'.format(i)].astype()
    globals()['grouped_{}'.format(i)]=globals()['df_{}'.format(i)][column_Name_list[1:]].groupby(globals()['df_{}'.format(i)]["Time"]).mean()
    
####
#### time을 index로 만들기 ### 필요없어졌습니다.
#for i in var_list:
    
    ####globals()['df_{}'.format(i)]=globals()['df_{}'.format(i)].set_index("Time")
    
    
##### 17개의 샘플들에서 시간대만 뽑아오자.
for i in var_list:
    globals()['df_{}'.format(i)]=globals()['grouped_{}'.format(i)].loc[overlapped_time_list_str]
    
    ####globals()['df2_{}'.format(i)]=globals()['df_{}'.format(i)].loc[overlapped_time_list_str]
    
    
#### unfolding 하기 17개의 샘플을 하나로 만들기.
for i in var_list:
    globals()['unfolded_df2_{}'.format(i)]=pd.DataFrame(globals()['df_{}'.format(i)].unstack())
    globals()['unfolded_df2_{}'.format(i)]=globals()['unfolded_df2_{}'.format(i)].T
    
    
    #unfolded_columns_Namelist = globals()['unfolded_df2_{}'.format(i)].columns

unfolded_columns_Namelist = list(unfolded_df2_14.columns)

##### 17개의 샘플들을 하나의 데이터프레임으로 합치기
X80 = pd.DataFrame(columns=unfolded_columns_Namelist)
#X=unfolded_df2_14
for i in var_list:
    X80=X80.append(globals()['unfolded_df2_{}'.format(i)], ignore_index=True)
X80 = X80.reset_index(drop=True)

####### Y 만들기
y80= pd.DataFrame({'y':[45.435,44.45,45.467,46.809,48.42,44.215,44.78,46.244,45.852,46.665,46.898,46.128,48.78,46.108,43.259,44.41,44.092]})


#### X와 y를 가로로 합치자. ####
data80 = pd.concat([X80,y80],axis=1)

#### CSV 파일로 저장합니다. ####
data80.to_csv('F:/JupyterLab/AdvancedIntelligence19/AI_Homework02/TSV ETCH/data80.csv')



###############################################


