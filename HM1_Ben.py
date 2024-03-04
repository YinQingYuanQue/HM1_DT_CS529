import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import chi2
from scipy.stats import mode
import math
#print ("Leo_test")

root_folder = 'HM1/'
train_file = 'train.csv'
#train_file = 'iris.csv'
test_file = 'test.csv'
Index = 'TransactionID'
#Index = ''
Target = 'isFraud'
#Target = 'species'
#Target_value = 'versicolor' # 1
Target_value = 1

train = pd.read_csv(root_folder + train_file)
#test = pd.read_csv(root_folder + test_file)
#print (train.columns)
train.shape

size = train.shape
Columns = train.columns.values
Columns = Columns[Columns != Index]
Categoric_features = ['ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2']
#Categoric_features = []
Numeric_features = np.setdiff1d(Columns, Categoric_features)
MissingData = pd.DataFrame(index = Columns)
for i in Columns:
  #fig, ax  = plt.subplots(figsize= (5,3))
  #plt.hist(train[i])
  # if i in Numeric_features:
  #       identify = 'Numeric'
  # else:
  #   identify = 'Categoric'
  # plt.title(f"{i:}, {identify}")
  # if i == 'isFraud':
  #       continue
  tmp_df = train[(train[i]=='NotFound') | (train[i].isna())] #test[(test[i]=='NotFound') | (test[i].isna())]
  MissingData.loc[i,0] = tmp_df.shape[0]
Categoric_features.append('Num_Cat')
print (Numeric_features)
print (Categoric_features)
MissingData


train_filter = train.copy() 
############# 1. remove missing records ############


############# 2. Use most frequency values ############
train_Impu = train.copy() 
#test_Impu = test.copy()
############# 2. Use most frequency values ############

for i in tqdm(Columns):
  train_filter = train_filter[train_filter[i]!='NotFound']
  train_Impu.loc[train_Impu[i]=='NotFound',i] = train_Impu[i].value_counts().idxmax()
  # if i != 'isFraud':
  #   test_Impu[test_Impu[i]=='NotFound'] = test[i].value_counts().idxmax()
print (train_filter.shape[0]/size[0]," of data left in train_filter")



def Info_Gain(Input_Dataframe, Split_feature, IG_methods, Num_Cat): #IG_methods = 'Gini'/ 'Entropy' / 'MisEr'

  length = len(Input_Dataframe)
  Fraud_arr = Input_Dataframe[Target]

  # propability of Fraud and not Fraud
  P_isF = np.sum(Fraud_arr == Target_value)/length
  P_notF = 1-P_isF

  Impu_root = 0
  Impu_leaves = [] #Impurity of each leaves
  Prop_leaves = [] #Propotion of each leaves
  IG_feature_Num = [] #IGs for numeric features

  ######### if Num_Cat == 'Mid' #########
  # This is used for recursive. 
  # When the Split_feature is a numeric feature, Info_Gain function will convert the numeric feature into a binary categorical feature (Line 74)
  # And then re-call the function Info_Gain(Input_Dataframe, Split_feature, IG_methods, 'Mid') , so the function will calculate the information gain for the numeric feature with the specific binary seperation
  ######### if Num_Cat == 'Mid' #########
  if Num_Cat == 'Cat' or Num_Cat == 'Mid':
    # Feature_classes: all unique classes, e.g. ['C','W','R', ...] for 'ProductCD'
    # P_unique_Fea: count for each classes, e.g. [100, 200, 100, ...]
    Feature_classes, P_unique_Fea = np.unique(Input_Dataframe[Split_feature], return_counts = True)

    if IG_methods == 'Gini':
      Impu_root = 1 - P_isF ** 2  - P_notF ** 2
      for i in Feature_classes:
        Sub_df = Input_Dataframe[Input_Dataframe[Split_feature] == i]
        Sub_len = len(Sub_df)
        Sub_P_isF = np.sum(Sub_df[Target] == Target_value)/Sub_len
        Sub_P_notF = 1-Sub_P_isF
        Impu_leaves.append(1 - Sub_P_isF ** 2  - Sub_P_notF ** 2)
        Prop_leaves.append(Sub_len/length)
    elif IG_methods == 'MisEr':
      Impu_root = 1 - max(P_isF,P_notF)
      for i in Feature_classes:
        Sub_df = Input_Dataframe[Input_Dataframe[Split_feature] == i]
        Sub_len = len(Sub_df)
        Sub_P_isF = np.sum(Sub_df[Target] == Target_value)/Sub_len
        Sub_P_notF = 1-Sub_P_isF
        Impu_leaves.append(1 - max(Sub_P_isF,Sub_P_notF))
        Prop_leaves.append(Sub_len/length)

    elif IG_methods == 'Entropy':
          
      ########## to deal with log(0) with 0 occurance ###########
      if P_isF == 0:
        Ent_P_isF = 0
      else:
        Ent_P_isF = P_isF * np.log2(P_isF)
      if P_notF == 0:
        Ent_P_notF = 0
      else:
        Ent_P_notF = P_notF * np.log2(P_notF)
      ########## to deal with log(0) with 0 occurance ###########

      Impu_root = -(Ent_P_notF + Ent_P_isF)
      for i in Feature_classes:
        Sub_df = Input_Dataframe[Input_Dataframe[Split_feature] == i]
        Sub_len = len(Sub_df)
        Sub_P_isF = np.sum(Sub_df[Target] == Target_value)/Sub_len
        Sub_P_notF = 1-Sub_P_isF
        if Sub_P_isF == 0:
          Ent_Sub_P_isF = 0
        else:
          Ent_Sub_P_isF = Sub_P_isF * np.log2(Sub_P_isF)
        if Sub_P_notF == 0:
          Ent_Sub_P_not_F = 0
        else:
          Ent_Sub_P_not_F = Sub_P_notF * np.log2(Sub_P_notF)
        Impu_leaves.append(-(Ent_Sub_P_isF + Ent_Sub_P_not_F))
        Prop_leaves.append(Sub_len/length)
    
  elif Num_Cat == 'Num':
    # this will automatically sort those unique values
    Sort_Input = np.unique(Input_Dataframe[Split_feature])
    Sort_Input = Sort_Input.astype(np.float32)

    # try every numeric values, add those information gain into the list "IG_feature_Num"   
    for i in Sort_Input:
      tmp_Input_Dataframe = Input_Dataframe.copy()
      # binary seperate the feature
      tmp_Input_Dataframe.loc[tmp_Input_Dataframe[Split_feature] <= i,Split_feature] = 0
      tmp_Input_Dataframe.loc[tmp_Input_Dataframe[Split_feature] > i,Split_feature] = 1
      IG_feature_Num.append(Info_Gain(tmp_Input_Dataframe,Split_feature, IG_methods, 'Mid'))

  Impu_root = np.array(Impu_root)
  Prop_leaves = np.array(Prop_leaves)

  # calculate the gain ratio if there are too much classes, this could reduce the breadth of the tree 
  if (Split_feature in Categoric_features) and (len(Feature_classes) > 200):
    P_unique_Fea = P_unique_Fea/sum(P_unique_Fea)
    log_P = np.log2(P_unique_Fea)
    SplitInfo = -np.matmul(P_unique_Fea,log_P.T)
    IG_feature =  float(f"{Impu_root - np.matmul(Impu_leaves,Prop_leaves.T):.5f}")
    IG_feature = IG_feature/SplitInfo
  # simplyt using the information gain
  elif (Split_feature in Categoric_features) or (Num_Cat == 'Mid'):
    IG_feature =  float(f"{Impu_root - np.matmul(Impu_leaves,Prop_leaves.T):.5f}")

  
  if Num_Cat == 'Num':
    IG_feature = max(IG_feature_Num)
    return [IG_feature, Sort_Input[IG_feature_Num.index(IG_feature)]] 
  elif Num_Cat == 'Mid':
    return IG_feature
  elif Num_Cat == 'Cat':
    return [IG_feature, -3.1415926] # -pi means it is a categorical feature
  

def Chi_test(Input_Dataframe, Split_feature, *Split_pts_Num):
  length = Input_Dataframe.shape[0]
  Fraud_arr = Input_Dataframe[Target]
  #print ('Pass line 3')
  # propability of Fraud and not Fraud
  P_isF = float(len(Fraud_arr[Fraud_arr == Target_value])/length)
  #print (P_isF)
  P_notF = 1-P_isF
  if (P_isF == 1) or (P_notF == 1):
    #print ('P_isF == 1')
    return 0
  Chi_sq = 0
  alpha = 0.05
  #print (Split_feature)
  if Split_feature in Categoric_features:
    #print ('in')
    
    Feature_classes = np.unique(Input_Dataframe[Split_feature])
    #print (Feature_classes)
    Degree_Freedom = (len(Feature_classes) - 1) * (2-1) # 2: T/F in isFraud
    #print (Degree_Freedom)
    for i in Feature_classes:
      Sub_df = Input_Dataframe[Input_Dataframe[Split_feature] == i]
      #print (Sub_df)
      Sub_len = len(Sub_df)
      Sub_P_isF = len(Sub_df[Sub_df[Target] == Target_value])/Sub_len
      #print (Sub_P_isF)
      Sub_P_notF = 1-Sub_P_isF
      Chi_sq = Chi_sq + Sub_len * ((Sub_P_isF - P_isF)**2/P_isF + (Sub_P_notF - P_notF)**2/P_notF)
    Chi_basic = chi2.ppf(1 - alpha, Degree_Freedom) - 5
    
  elif Split_feature in Numeric_features:
    Degree_Freedom = 1*1
    tmp_Input_Dataframe = Input_Dataframe.copy()
    #print (tmp_Input_Dataframe)
    #print (Split_pts_Num)
    Split_pts_Num = Split_pts_Num
    #tmp_Input_Dataframe[Split_feature] < 2
    tmp_Input_Dataframe.loc[tmp_Input_Dataframe[Split_feature] <= Split_pts_Num,'Num_Cat'] = 0
    tmp_Input_Dataframe.loc[tmp_Input_Dataframe[Split_feature] > Split_pts_Num,'Num_Cat'] = 1
    Chi_sq = Chi_test(tmp_Input_Dataframe,'Num_Cat')
    Chi_basic = chi2.ppf(1 - alpha, Degree_Freedom) - 5
  
  if Split_feature != 'Num_Cat':
    if Chi_sq > Chi_basic:
      #print (Chi_sq, Chi_basic)
      return 1  # good result, expanding
    else:
      #print (Chi_sq, Chi_basic)
      return 0  # bad result, not expanding
  if Split_feature == 'Num_Cat':
      return Chi_sq


class TreeNode:
    # Self_Column: feature
    # Sub_branches: line connecting two nodes
    def __init__(self, Self_Column = None, Sub_branches = None, Target = None, Sub_childs = None, Parent_Col = None):
        self.Self_Column = Self_Column 
        self.Sub_branches = Sub_branches if Sub_branches is not None else []
        self.Target = Target
        self.Sub_childs = Sub_childs if Sub_childs is not None else []
        self.Parent_Col = Parent_Col
        

    def is_leaf_node(self):
        return self.Target is not None
    
    def set_values(self, column, branch):
        self.Self_Column = column
        self.Sub_branches = branch
    
    def set_Target(self, target):
        self.Target = target
    
    def add_child(self, node):
        self.Sub_childs.append(node)
        node.set_parent(self.Self_Column)

    def get_branch(self):
        #print ('Column, Branches, Target')
        #print (self.Self_Column, self.Sub_branches, self.Target)
        return self.Sub_branches
    
    def get_child(self):
        return self.Sub_childs
    
    def get_column(self):
        return self.Self_Column
    
    def get_target(self):
        return self.Target
    
    def set_parent(self, parent_col):
        self.Parent_Col = parent_col
        

    
    
def build_DT(Input_Dataframe, total_Col = Columns, IG_methods = 'Gini', depth = 0):
    #tmp_Col = total_Col
    #IG_methods = 'Gini'/ 'Entropy' / 'MisEr'
    IG_method = IG_methods
    max_depth = 5
    Root_node = TreeNode()
    self_col = Input_Dataframe.columns.values
    depth = depth
    
    IG_columns = pd.DataFrame(columns= ['Column', 'IG', 'SplitPts'])
    n = 0
    for i in self_col:
        #print (i)
        if i == Target or i == Index or not (i in total_Col):
            continue
        elif i in Categoric_features:
            [tmp_IG, Split_pts] = Info_Gain(Input_Dataframe, i, IG_method, 'Cat')
        elif i in Numeric_features:
            [tmp_IG, Split_pts] = Info_Gain(Input_Dataframe, i, IG_method, 'Num')
        #print (i, total_Col, 'i in total_Col', i in total_Col)
        IG_columns.loc[n,'Column'] = i
        IG_columns.loc[n,'IG'] = tmp_IG
        IG_columns.loc[n,'SplitPts'] = Split_pts
        n = n + 1

    #print (IG_columns)
    Best_IG = IG_columns['IG'].max()
    [Best_Column, Split_Pts] = IG_columns.loc[IG_columns['IG'] == Best_IG, ['Column', 'SplitPts']].values[0]
    #print (Best_Column,Split_Pts)
    
    if Best_Column in Categoric_features:
        Root_node.set_values(Best_Column, np.unique(Input_Dataframe[Best_Column]))
        
    elif Best_Column in Numeric_features:
        Root_node.set_values(Best_Column, Split_Pts)
    if (not Chi_test(Input_Dataframe, Best_Column, Split_Pts)):
        Root_node.set_Target(Input_Dataframe[Target].mode()[0])
        #print ('fail chi_test:', depth, Best_Column)
        #print ('max_depth', max_depth, depth == max_depth)
        return Root_node
    else:
        depth = depth + 1

        #print ('pass chi_test:', depth)
        if depth < max_depth:
            #print ('in, Self_Col:', Root_node.get_column())
            #print ('in, Self_branch:', Root_node.get_branch())
            #print ('in, Self_branch_len:', len(Root_node.get_branch()))
            if (Root_node.Self_Column in Categoric_features) and (Root_node.Self_Column in total_Col):                
                for i in Root_node.Sub_branches:
                    #print ('Current Sub_Branch = ',i)
                    Sub_DF = Input_Dataframe[Input_Dataframe[Root_node.Self_Column] == i]    
                    tmp_Col = total_Col[total_Col != i]               
                    Root_node.add_child(build_DT(Sub_DF, tmp_Col, IG_method, depth))
                
                #print ("cat:", depth)
            elif (Root_node.Self_Column in Numeric_features) and (Root_node.Self_Column in total_Col):
                
                Sub_DF_left = Input_Dataframe[Input_Dataframe[Root_node.Self_Column] <= Root_node.Sub_branches]
                tmp_Col = total_Col[total_Col != i]
                Root_node.add_child(build_DT(Sub_DF_left, tmp_Col, IG_method, depth))
                Sub_DF_right = Input_Dataframe[Input_Dataframe[Root_node.Self_Column] > Root_node.Sub_branches]
                tmp_Col = total_Col[total_Col != i]
                Root_node.add_child(build_DT(Sub_DF_right, tmp_Col, IG_method, depth))
                
                
                #print ("num:", depth)
        else:
            #print ("depth maximum")
            Root_node.set_Target(Input_Dataframe[Target].mode()[0])
        return Root_node
    

def DT_Predict(Input_Dataframe,Tree_node):
    tmp_node = Tree_node
    potential_predict = np.array([])
    list_TF = 0
    while not tmp_node.is_leaf_node():
        tmp_col = tmp_node.get_column()
        #tmp_col = tmp_node.Self_Column
        data_val = Input_Dataframe[tmp_col].values[0]
        branches = tmp_node.get_branch()
        #branches = tmp_node.Sub_branches
        
        if tmp_col in Categoric_features:
            #print ("cat:", tmp_col)
            #print (data_val)
            sub_node_idx = np.where(branches == data_val)[0]
            if sub_node_idx.shape[0] > 0:
                sub_node_idx = sub_node_idx[0]
                list_TF = 0
            else:
                sub_node_idx = -1
                list_TF = 1
                
        elif tmp_col in Numeric_features:
            list_TF = 0
            #print ("num:", tmp_col)
            if data_val <= branches:
                sub_node_idx = 0
            else:
                sub_node_idx = 1
        
        child_nodes = tmp_node.get_child()
        #print (sub_node_idx)
        #print (child_nodes)
        if sub_node_idx == -1:
            for i in branches:
                tmp_Dataframe = Input_Dataframe.copy()
                sub_node_idx = np.where(branches == i)[0][0]
                tmp_node = child_nodes[sub_node_idx]
                tmp_Dataframe[tmp_col] = i
                potential_predict = np.append(potential_predict, DT_Predict(tmp_Dataframe, tmp_node))
        else:
            #print (tmp_node.get_column(), len(tmp_node.get_child()), tmp_node.is_leaf_node(), tmp_node.get_target())
            tmp_node = child_nodes[sub_node_idx]
    #print (sub_node_idx)
    if list_TF == 1:
        return potential_predict
    else:
        return tmp_node.Target
    

def tree_df(root_node):

    queue = [(root_node,0,0)]
    list_node = []
    while queue:
        current_node, depth, sub_branch = queue.pop(0)
        current_col_name = current_node.get_column()
        current_branches = current_node.get_branch()
        current_childs = current_node.get_child()
        childs_name = []
        for i in range(len(current_childs)):
            childs_name.append(current_childs[i].get_column())
        list_node.append((current_col_name,current_branches,childs_name,depth, current_node.get_target(),current_node.Parent_Col,sub_branch))
        #print (list_node)
        tmp_child = current_node.get_child()
        if len(tmp_child) > 0:
            if current_col_name in Categoric_features:
                for i in range(len(tmp_child)):
                    queue.append((tmp_child[i],depth+1,current_branches[i]))
            elif current_col_name in Numeric_features:
                queue.append((tmp_child[0],depth+1,current_branches))
                queue.append((tmp_child[1],depth+1,current_branches+1))

    tree_df = pd.DataFrame(list_node,columns = ['Column_name','Branches','Child_Columns','Depth','Target','Parent_Col','Parent_Branch'])
    return tree_df




def Predict_df(Input_Dataframe,tree_df, current_index = 0):

    potential_predict = np.array([])
    list_TF = 0
    current_index = current_index
    while pd.isna(tree_df.loc[current_index,'Target']):
        
        tmp_col = tree_df.loc[current_index,'Column_name']
        data_val = Input_Dataframe[tmp_col].values[0]
        branches = tree_df.loc[current_index,'Branches']
        
        if tmp_col in Categoric_features:
            #print ("cat:", tmp_col)
            #print (data_val)
            sub_node_idx = np.where(branches == data_val)[0]
            if sub_node_idx.shape[0] > 0:
                sub_node_idx = sub_node_idx[0]
                tmp_branch = branches[sub_node_idx]
                list_TF = 0
            else:
                sub_node_idx = -1
                list_TF = 1
                
        elif tmp_col in Numeric_features:
            list_TF = 0
            #print ("num:", tmp_col)
            if data_val <= branches:
                sub_node_idx = 0
                tmp_branch = branches
            else:
                sub_node_idx = 1
                tmp_branch = branches + 1
        
        child_nodes_name = tree_df.loc[current_index,'Child_Columns']

        if sub_node_idx == -1:
            for i in branches:
                tmp_Dataframe = Input_Dataframe.copy()
                #print ('i=', i)
                #print (branches,np.where(branches == i)[0])
                branch_idx = np.where(branches == i)[0][0]
                
                Child_Name = child_nodes_name[branch_idx]
                current_index = tree_df[(tree_df['Column_name'] == Child_Name) & (tree_df['Parent_Col'] == tmp_col) & (tree_df['Parent_Branch'] == i)].index[0]
                
                tmp_Dataframe[tmp_col] = i
                potential_predict = np.append(potential_predict, Predict_df(tmp_Dataframe, tree_df, current_index))
        else:
            #print (tmp_node.get_column(), len(tmp_node.get_child()), tmp_node.is_leaf_node(), tmp_node.get_target())

            current_index = tree_df[(tree_df['Column_name'] == child_nodes_name[sub_node_idx]) & (tree_df['Parent_Col'] == tmp_col) & (tree_df['Parent_Branch'] == tmp_branch)].index[0]
    #print (sub_node_idx)
    if list_TF == 1:
        return potential_predict
    else:
        return tree_df.loc[current_index,'Target']

#tree_df(tree_node)
    
#######################################################################################################
Tree_no = 2
#test_test = train_filter.sample(frac = 1)[0:4]
All_predict = []
All_observe = []


for ii in range(Tree_no):
    train_filter = train_filter.sample(frac = 1)
    positive = train_filter[train_filter['isFraud'] == 1]
    negative = train_filter[train_filter['isFraud'] == 0]
    pos_size = positive.shape
    neg_size = negative.shape

    test_positive = positive[0:int(pos_size[0])]
    test_negative = negative[0:int(neg_size[0])]

    train_positive = positive[int(pos_size[0]):int(pos_size[0])]
    train_negative = negative[int(neg_size[0]):int(neg_size[0])]

    pool_train = pd.concat([train_positive, train_negative])
    pool_test = pd.concat([test_positive, test_negative])
    pool_test = pool_test.sample(frac = 1)
    exp_test = pool_test[0:int(len(pool_test))]
    observe = []



    predict = []    
    np.random.shuffle(Columns)
    tmp_total_Col = Columns[0:20]

    pool_train = pool_train.sample(frac = 1)
    exp_train = pool_train[0:int(len(pool_train)*0.9)]
    
    print (exp_train.shape)

    #IG_methods = 'Gini'/ 'Entropy' / 'MisEr'
    tree_node = build_DT(exp_train, total_Col= tmp_total_Col, IG_methods = 'Gini')
    T_Df = tree_df(tree_node)
    T_Df.to_pickle('Tree_' + str(ii) + '.pkl')
    #T_Df.to_csv('Tree_' + str(i) + '.csv')
    print (str(ii) + 'th DT')
    for i in tqdm(range(exp_test.shape[0])):
        tmp_dataframe = pd.DataFrame(exp_test.iloc[i]).T
        #(type(tmp_dataframe))
        #print (tmp_dataframe)
        target_predict = DT_Predict(tmp_dataframe,tree_node)
        if not isinstance(target_predict, np.ndarray):
            predict.append(target_predict)
        else:
            predict.append(mode(target_predict).mode[0])
    observe = exp_test[Target]

    All_predict.append(predict)
    All_observe.append(observe)

    Confuse_Matrix = pd.DataFrame(0,columns = ['Predict_T','Predict_F'],index= ['Observe_T','Observe_F'])
    for i in tqdm(range(len(predict))):
        if predict[i] == 1 and observe.iloc[i] == 1:
            Confuse_Matrix.loc['Observe_T','Predict_T'] = Confuse_Matrix.loc['Observe_T','Predict_T'] + 1
        elif predict[i] == 1 and observe.iloc[i] == 0:
            Confuse_Matrix.loc['Observe_F','Predict_T'] = Confuse_Matrix.loc['Observe_F','Predict_T'] + 1
        elif predict[i] == 0 and observe.iloc[i] == 1:
            Confuse_Matrix.loc['Observe_T','Predict_F'] = Confuse_Matrix.loc['Observe_T','Predict_F'] + 1
        elif predict[i] == 0 and observe.iloc[i] == 0:
            Confuse_Matrix.loc['Observe_F','Predict_F'] = Confuse_Matrix.loc['Observe_F','Predict_F'] + 1
    Recall_P = Confuse_Matrix.loc['Observe_T','Predict_T'] / sum(Confuse_Matrix.loc['Observe_T'])
    Recall_F = Confuse_Matrix.loc['Observe_F','Predict_F'] / sum(Confuse_Matrix.loc['Observe_F'])
    Balanced_Acc = (Recall_P + Recall_F)/2
    print("Balanced_Acc:", Balanced_Acc)
    Confuse_Matrix.to_csv('Confuse_Matrix' + str(ii))