from unicodedata import name
import pandas as pd
import os

path_binhthuong="Ket qua thu nghiem/Ket qua thu nghiem/che_do_binh_thuong"
path_suco="Ket qua thu nghiem/Ket qua thu nghiem/Su_co"

# class
list_class_binhthuong = os.listdir(path_binhthuong)

#name_class
name_column=[]
values_column=[]

for l_c in list_class_binhthuong:
    path_folder = os.path.join(path_binhthuong,l_c)
    name_class = os.listdir(path_folder)
    count_folder=0
    for l_f in name_class:
        os_name = os.path.join(path_folder,l_f)
        name_diem = os.listdir(os_name)
        for l_d in name_diem:
            if  l_d.split(".")[1]=="csv":
                count_folder = count_folder + 1
                df = pd.read_csv(os.path.join(os_name,l_d), sep=";",encoding='utf-16')
                #print(os.path.join(os_name,l_d))
                #print(df)
                name_column = df["Trend 1 Time"].tolist()
                array = df["Trend 1 ValueY"].tolist()
                #print(len(array))
                if len(array) == 240:
                    array.pop()
                    array.pop()
                    #print("doan sang2")
                if len(array) == 239:
                    array.pop()
                    #print("doan sang2")
                if len(array) > 237:
                    for index in range(237,len(array)):
                        array.pop()
                    #array.pop()
                #print("LEN: ",len(array))
                array.append(0)
                
                values_column.append(array)
                #print(array)
                #break
    print(l_c," : ",count_folder)
print("len binh thương: ",len(values_column))
list_class_suco = os.listdir(path_suco)
for l_c in list_class_suco:
    path_folder = os.path.join(path_suco,l_c)
    name_class = os.listdir(path_folder)
    count_folder=0
    for l_f in name_class:
        os_name = os.path.join(path_folder,l_f)
        name_diem = os.listdir(os_name)
        for l_d in name_diem:
            if  l_d.split(".")[1]=="csv":
                count_folder = count_folder + 1
                df = pd.read_csv(os.path.join(os_name,l_d), sep=";",encoding='utf-16')
                # print(os.path.join(os_name,l_d))
                # print(df)
                array = df["Trend 1 ValueY"].tolist()
                #print(len(array))
                if len(array) == 240:
                    array.pop()
                    array.pop()
                    #print("doan sang2")
                if len(array) == 239:
                    array.pop()
                    #print("doan sang2")
                if len(array) > 237:
                    for index in range(237,len(array)):
                        #print("doan sang")
                        array.pop()
                #print("LEN: ",len(array))
                array.append(1)
                values_column.append(array)
                #break
    print(l_c," : ",count_folder)
name_column.pop()
name_column.append("class")
print("len total: ",len(values_column))
df = pd.DataFrame(values_column,columns = name_column)
df.to_csv("data.csv")