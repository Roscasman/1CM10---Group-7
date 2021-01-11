# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:40:27 2020

@author: s149562
"""

# Import libaries: 
import numpy as np
import pandas as pd
import operator
import random
import matplotlib.pyplot as plt
import math
import itertools
import gurobipy as gp
from gurobipy import GRB
import time
import timeit


# Load & sort data:
df = pd.read_csv('6M140.csv')
header = list(df.columns.values)
number_of_operations = header[1] 
number_of_machines = df.at[0, number_of_operations] 
number_of_toolsets = df.at[1, number_of_operations] 
capacity_per_machine = int(df.at[2, number_of_operations] )
# number_of_operations = int
# number_of_machines = int
# number_of_toolsets = int
# capacity_per_machine = int
df = df.drop(range(5))
df.columns = ['Job_i', 'operation_j', 'Releasetime_r_ij', 'Processing_time_p_ij', 'Due_date_d_ij','Tool_set_T_ij','Tool_set_size_Φ_T_ij']
df = df.reset_index(drop=True)
df = df.astype({'Job_i':int, 'operation_j':int, 'Releasetime_r_ij':float, 'Processing_time_p_ij':float, 'Due_date_d_ij':float,'Tool_set_T_ij':int,'Tool_set_size_Φ_T_ij':int})




class Operation:

    # Initializer / Instance Attributes
    def __init__(self,myid,myoperation,myreleasetime,myprocessingtime,myduedate,mytoolset,mytoolsetsize):
        self.job = myid
        self.operation = myoperation
        self.releasetime = myreleasetime
        self.processingtime=myprocessingtime
        self.duedate=myduedate
        self.toolset=mytoolset
        self.toolsetsize=mytoolsetsize
        self.tool_switch=0
        self.endtime=0
        
    def reset_operation(self):
        self.tool_switch=0
        self.endtime=0
    



class Machine:
    def __init__(self,mytoolsets,mycapacity):
        self.toolsets_and_sizes=mytoolsets
        self.capacity=int(mycapacity)
        self.time_finished=0
        self.sum_processingtimes = 0
    
    def add_job(self, job_end_time):
        self.time_finished=job_end_time
    
    
    def change_capacity(self, toolsetsize):
        self.capacity=int(toolsetsize)
        
    def add_toolset(self, toolset_and_size):
        self.toolsets_and_sizes.append((toolset_and_size))
    
    def Reset_machine(self):
        self.toolsets_and_sizes=[]
        self.capacity=capacity_per_machine
        self.time_finished=0
        self.sum_processingtimes = 0
        
        
        
Jobb_vector_init=[]
for i in range(0, len(df)):

    Jobb_vector_init.append("Operation_"+str(df.Job_i[i])+"_"+str(df.operation_j[i]))

list_of_job_vector=(())
i=0   
for a in Jobb_vector_init: 
    a=Operation(df.Job_i[i],df.operation_j[i],df.Releasetime_r_ij[i], df.Processing_time_p_ij[i],df.Due_date_d_ij[i], df.Tool_set_T_ij[i], df.Tool_set_size_Φ_T_ij[i])
    b="Operation_"+str(df.Job_i[i])+"_"+str(df.operation_j[i])
    list_of_job_vector=list_of_job_vector+((a,b))
    i=i+1

objectnames=Jobb_vector_init
Operationdictionairy={}
i=0
for name in objectnames:
        Operationdictionairy[name]=Operation(df.Job_i[i],df.operation_j[i],df.Releasetime_r_ij[i], df.Processing_time_p_ij[i],df.Due_date_d_ij[i], df.Tool_set_T_ij[i], df.Tool_set_size_Φ_T_ij[i])
        i=i+1

machinelist=[]
for x in range (1,(int(number_of_machines)+1)):
     z="Machine_"+str(x)
     machinelist.append(z)
Machinedictionairy={}
for machine in machinelist:
    Machinedictionairy[machine]=Machine([], int(capacity_per_machine))
    




    

# Set paramaters accoring to CH 7:
B = 1
N_p = 100
Y_2 = 0.10
P_u = 0.01
P_s = 0.01
Y_1 = 0.20
Weight_tardiness = 1
Weight_tool_setup = 1
maxtime = 3600
G_c = 20



# Initialization:
#Empty lists to store the created PH and SH vectors
list_PH_job_vectors = []
list_PH_machine_vectors = []

## Practitioner heuristic: 
input_file = pd.read_csv('6M140.csv',
index_col=False,
keep_default_na=True
)

header = list(input_file.columns.values)

number_of_operations = header[1]
number_of_machines = input_file.at[0, number_of_operations]
number_of_toolsets = input_file.at[1, number_of_operations]
capacity_per_machine = int(input_file.at[2, number_of_operations])

list_capacity = [int(capacity_per_machine)] * int(number_of_machines)


#binary array om bij te houden welke toolset aan welke machine is toegekend
binary_array = np.zeros([int(number_of_toolsets), int(number_of_machines)], dtype = int)
array_toolsizes = np.zeros([int(number_of_toolsets), int(number_of_machines)], dtype = int)

input_file = pd.read_csv("6M140.csv",
                         header = 5)

set_operations=input_file
set_operations=set_operations.rename(columns={"Release time": "Release_time","Processing time": "Processing_time", "Due date": "Due_date", "Tool set": "Toolset", "Tool set size": "Toolsetsize"}, inplace = True)

machine_with_number_list=[]
for y in range (1, int(number_of_machines) + 1):
    machine_with_number_list.append('Machine_' + str(y))
    
sequence_of_jobs = {i:[] for i in machine_with_number_list}

a_m_m = {i:0 for i in machine_with_number_list}

τ = 1
w_d = 1
w_s = 1

# einde input part practitioner heuristic
#Execute practitioner heuristic
import Practicioner_heureustic_eindresultaat
run_PH = Practicioner_heureustic_eindresultaat.Practitioner_heuristic(input_file, binary_array, list_capacity)
Practicioner_job_vector = run_PH[2]
Practicioner_machine_vector = run_PH[1]
objective_value_PH = run_PH[0]
list_PH_job_vectors.append(Practicioner_job_vector)
list_PH_machine_vectors.append(Practicioner_machine_vector)




#Simple heuristic
#initialisation

def sort_jobs(job):
    return Operationdictionairy[job].duedate    

def unique_toolsets_and_sizes_calculator(Jobb_vector_init):
    toolsets=[]
    number_toolsets=0
    size_of_toolsets=0
    for job in Jobb_vector_init:
        if Operationdictionairy[job].toolset not in toolsets:
            toolsets.append (Operationdictionairy[job].toolset)
            number_toolsets=number_toolsets+1
            size_of_toolsets=size_of_toolsets+Operationdictionairy[job].toolsetsize
    return (size_of_toolsets/number_toolsets)


for x in range(9):
    # Load & sort data:
    df = pd.read_csv('6M140.csv')
    header = list(df.columns.values)
    number_of_operations = header[1] 
    number_of_machines = df.at[0, number_of_operations] 
    number_of_toolsets = df.at[1, number_of_operations] 
    capacity_per_machine = df.at[2, number_of_operations] 
    # number_of_operations = int
    # number_of_machines = int
    # number_of_toolsets = int
    # capacity_per_machine = int
    df = df.drop(range(5))
    df.columns = ['Job_i', 'operation_j', 'Releasetime_r_ij', 'Processing_time_p_ij', 'Due_date_d_ij','Tool_set_T_ij','Tool_set_size_Φ_T_ij']
    df = df.reset_index(drop=True)
    df = df.astype({'Job_i':int, 'operation_j':int, 'Releasetime_r_ij':float, 'Processing_time_p_ij':float, 'Due_date_d_ij':float,'Tool_set_T_ij':int,'Tool_set_size_Φ_T_ij':int})
    
    Jobb_vector_init=[]
    for i in range(0, len(df)):
    
        Jobb_vector_init.append("Operation_"+str(df.Job_i[i])+"_"+str(df.operation_j[i]))
    
    list_of_job_vector=(())
    i=0   
    for a in Jobb_vector_init: 
        a=Operation(df.Job_i[i],df.operation_j[i],df.Releasetime_r_ij[i], df.Processing_time_p_ij[i],df.Due_date_d_ij[i], df.Tool_set_T_ij[i], df.Tool_set_size_Φ_T_ij[i])
        b="Operation_"+str(df.Job_i[i])+"_"+str(df.operation_j[i])
        list_of_job_vector=list_of_job_vector+((a,b))
        i=i+1
    
    objectnames=Jobb_vector_init
    Operationdictionairy={}
    i=0
    for name in objectnames:
            Operationdictionairy[name]=Operation(df.Job_i[i],df.operation_j[i],df.Releasetime_r_ij[i], df.Processing_time_p_ij[i],df.Due_date_d_ij[i], df.Tool_set_T_ij[i], df.Tool_set_size_Φ_T_ij[i])
            i=i+1
    
    machinelist=[]
    for x in range (1,(int(number_of_machines)+1)):
         z="Machine_"+str(x)
         machinelist.append(z)
    Machinedictionairy={}
    for machine in machinelist:
        Machinedictionairy[machine]=Machine([], int(capacity_per_machine))
    
    #End of initialisation of simple heuristic
    
    import Very_simple_algorithm
    run_Very_simple_algorithm=Very_simple_algorithm.Simple_algorithm(Jobb_vector_init)
    Very_simple_algorithm_job_vector=run_Very_simple_algorithm[0]
    Very_simple_algorithm_machine_vector=run_Very_simple_algorithm=[1]
    list_PH_job_vectors.append(Very_simple_algorithm_job_vector)
    list_PH_machine_vectors.append(Very_simple_algorithm_machine_vector)



## Random initialization:
job_vector=Jobb_vector_init
def Random_initialization(job_vector, machinelist):
    job_vectors_list_parents=[]
    machine_vectors_list_parents=[]
    for x in range (0, N_p - 10): #Minus 1, because practitioner heuristic is also added
        new=random.sample(job_vector, len(job_vector))
        job_vectors_list_parents.append(new)
        new_machine_vector=[]
        for y in range (0, len(job_vector)):
            b=np.random.randint(0, len(machinelist))
            new_machine_vector.append(machinelist[b])
        machine_vectors_list_parents.append(new_machine_vector)
    return (job_vectors_list_parents, machine_vectors_list_parents)



def Reset_classes():
    for machine in machinelist:
        Machinedictionairy[machine].Reset_machine()
    for job in job_vector:
        Operationdictionairy[job].reset_operation()

        

def job_vector_re_operation_sorter(job_vector): 

    a=-1
    for job in job_vector:
        a=a+1
        if Operationdictionairy[job].operation>1:
            c=Operationdictionairy[job].job
            switched_1=job
            b=-1
            for job_2 in job_vector:
                b=b+1
                if (Operationdictionairy[job_2].job==c) and ((Operationdictionairy[job_2].operation)==(Operationdictionairy[job].operation-1)):
                    switched_2=job_2
                    job_vector[min(a,b)]=switched_2
                    job_vector[max(a,b)]=switched_1
               
    return (job_vector)
    
Very_simple_algorithm_job_vector=job_vector_re_operation_sorter(Very_simple_algorithm_job_vector)                                        
#Select 2 parents for crossovers --> use tournament selection             
#In the tournament selection, a number of S_T chromosomes are first randomly chosen from the population where
#S_T = γ1 * N_p and γ1 is a percent of the population size N_p and called tournament selection rate (0 < γ1 ≤1).
#The chromosome with the best (lowest) fitness value among them is then selected as one parent chromosome.
#The other parent can also be determined in the same manner.                    
            
        
        
def two_point_crossover(machine_vector_1, machine_vector_2): ## heb nu nog dat parents random worden gekozen, maar zou later wss moeten worden aangepast
    
    cut_1=np.random.randint(0, len(machine_vector_1)-1)
    cut_2=np.random.randint(cut_1, len(machine_vector_1))
    new_machine_vector_1=machine_vector_1[0:cut_1]+machine_vector_2[cut_1:cut_2]+machine_vector_1[cut_2:]
    new_machine_vector_2=machine_vector_2[0:cut_1]+machine_vector_1[cut_1:cut_2]+machine_vector_2[cut_2:]
    
    return (new_machine_vector_1, new_machine_vector_2)

  
    
def APMX_crossover(job_vector1,job_vector2): ## weer random 2 parents gepakt maar dit zou later wel aangepast moeten worden

            
    mapping_tuple_parent1=[]
    b=0    
    for x in job_vector1:
        mapping_tuple_parent1.append((x, b))
        b=b+1
    TP_1=[]
    for y in mapping_tuple_parent1:
        TP_1.append(y[1])
    TP_2=[]
    for x in job_vector2:
        for y in mapping_tuple_parent1:
            if x==y[0]:
                TP_2.append(y[1])
    ## select substrings
    cut_1=np.random.randint(0, len(job_vector1)-1)
    cut_2=np.random.randint(cut_1+1, len(job_vector1))
    PO_1=TP_1[:cut_1]+TP_2[cut_1:cut_2]+TP_1[cut_2:]
    PO_2=TP_2[:cut_1]+TP_1[cut_1:cut_2]+TP_2[cut_2:]
    mapping_between_substrings=[]
    for x in range(0, len(TP_2[cut_1:cut_2])):
        mapping_between_substrings.append ((TP_2[cut_1:cut_2][x],TP_1[cut_1:cut_2][x]) ) 

    is_in_1=[]
    is_in_2=[]
    for w in mapping_between_substrings:
        is_in_1.append(w[0])
        is_in_2.append(w[1])
    mapping_between_substrings_v2=[]
    is_in_1_not_in_2=[]
    is_in_2_not_in_1=[]
    for x in is_in_1:
        if x not in is_in_2:
            is_in_1_not_in_2.append(x)
    for x in is_in_2:
        if x not in is_in_1:
            is_in_2_not_in_1.append(x)
    for z in range (0, len(is_in_1_not_in_2)):
        mapping_between_substrings_v2.append((is_in_1_not_in_2[z], is_in_2_not_in_1[z]))
                       
                
    c = 0        
    while c < cut_1:
        for x in PO_1[:cut_1]:
            for y in mapping_between_substrings_v2:
                if x == y[0]:
                    PO_1[c] = y[1]
            c += 1
    
    c = cut_2        
    while c <= len(PO_1):
        for x in PO_1[cut_2:]:
            for y in mapping_between_substrings_v2:
                if x == y[0]:
                    PO_1[c] = y[1]
            c +=1
    
    c = 0        
    while c < cut_1:
        for x in PO_2[:cut_1]:
            for y in mapping_between_substrings_v2:
                if x == y[1]:
                    PO_2[c] = y[0]
            c += 1
    
    c = cut_2        
    while c <= len(PO_2):
        for x in PO_2[cut_2:]:
            for y in mapping_between_substrings_v2:
                if x == y[1]:
                    PO_2[c] = y[0]
            c +=1

    O_1=[]
    for x in PO_1:
       for z in mapping_tuple_parent1:
           if x==z[1]:
               O_1.append(z[0])
    O_2=[]           
    for x in PO_2:
       for z in mapping_tuple_parent1:
           if x==z[1]:
               O_2.append(z[0])
#    print(O_1, O_2)
               


    return (O_1, O_2)
    
 
    
def Extended_APMX_crossover(job_vector1, job_vector2): #Use tournament selection to select 2 parents, then use the jobvectors of those parents to initialise the crossover
        
    print('before')
    for x in Jobb_vector_init:
        if x not in job_vector1:
            print('NO!')
            break
    for x in Jobb_vector_init:
        if x not in job_vector2:
            print('NO!')
            break
    mapping_tuple_parent1 = []
    b = 0
    for x in job_vector1:
        mapping_tuple_parent1.append((x, b))
        b = b + 1
    TP_1 = []
    for y in mapping_tuple_parent1:
        TP_1.append(y[1])
    TP_2 = []
    for x in job_vector2:
        for y in mapping_tuple_parent1:
            if x == y[0]:
                TP_2.append(y[1])
                break
    ## select substrings
    cut_1 = np.random.randint(0, len(job_vector1) - 1)
    cut_2 = np.random.randint(cut_1 + 1, len(job_vector1))
    PO_1 = TP_1[:cut_1] + TP_2[cut_1:cut_2] + TP_1[cut_2:]
    PO_2 = TP_2[:cut_1] + TP_1[cut_1:cut_2] + TP_2[cut_2:]
    mapping_between_substrings = []
    for x in range(0, len(TP_2[cut_1:cut_2])):
        mapping_between_substrings.append ((TP_2[cut_1:cut_2][x], TP_1[cut_1:cut_2][x]))  
    mapping_between_substrings_v2 = []    
    is_in_1=[]
    is_in_2=[]
    for w in mapping_between_substrings:
        is_in_1.append(w[0])
        is_in_2.append(w[1])
    mapping_between_substrings_v2=[]
    is_in_1_not_in_2=[]
    is_in_2_not_in_1=[]
    for x in is_in_1:
        if x not in is_in_2:
            is_in_1_not_in_2.append(x)
    for x in is_in_2:
        if x not in is_in_1:
            is_in_2_not_in_1.append(x)
    for z in range (0, len(is_in_1_not_in_2)):
        mapping_between_substrings_v2.append((is_in_1_not_in_2[z], is_in_2_not_in_1[z]))
    print(mapping_between_substrings)
    print(mapping_between_substrings_v2)
    c = 0        
    while c < cut_1:
        for x in PO_1[:cut_1]:
            for y in mapping_between_substrings_v2:
                if x == y[0]:
                    PO_1[c] = y[1]
            c += 1
    
    c = cut_2        
    while c <= len(PO_1):
        for x in PO_1[cut_2:]:
            for y in mapping_between_substrings_v2:
                if x == y[0]:
                    PO_1[c] = y[1]
            c +=1
    
    c = 0        
    while c < cut_1:
        for x in PO_2[:cut_1]:
            for y in mapping_between_substrings_v2:
                if x == y[1]:
                    PO_2[c] = y[0]
            c += 1
    
    c = cut_2        
    while c <= len(PO_2):
        for x in PO_2[cut_2:]:
            for y in mapping_between_substrings_v2:
                if x == y[1]:
                    PO_2[c] = y[0]
            c +=1
  
    O_1 = []
    for x in PO_1:
       for z in mapping_tuple_parent1:
           if x == z[1]:
               O_1.append(z[0])
    O_2 = []           
    for x in PO_2:
       for z in mapping_tuple_parent1:
           if x == z[1]:
               O_2.append(z[0])
    
    print('after')
    for x in Jobb_vector_init:
        if x not in O_1:
            print('NO!')
            break
    for x in Jobb_vector_init:
        if x not in O_2:
            print('NO!')
            break        

    #Extended part
    #RO_1
    before_cut = []
    after_cut = []
    RO_1 = []
    for job in O_1[:cut_1]:
        duedate = Operationdictionairy[job].duedate
        before_cut.append((job, duedate))
    before_cut.sort(key = lambda x: x[1])
    for job in before_cut:
        RO_1.append(job[0])
    
    for job in O_1[cut_1:cut_2]:
        RO_1.append(job)
    
    for job in O_1[cut_2:]:
        duedate = Operationdictionairy[job].duedate
        after_cut.append((job, duedate))
    after_cut.sort(key = lambda x: x[1])
    for job in after_cut:
        RO_1.append(job[0])
    
    #RO_2
    before_cut = []
    after_cut = []
    RO_2 = []
    for job in O_2[:cut_1]:
        duedate = Operationdictionairy[job].duedate
        before_cut.append((job, duedate))
    before_cut.sort(key = lambda x: x[1])
    for job in before_cut:
        RO_2.append(job[0])
    
    for job in O_2[cut_1:cut_2]:
        RO_2.append(job)
    
    for job in O_2[cut_2:]:
        duedate = Operationdictionairy[job].duedate
        after_cut.append((job, duedate))
    after_cut.sort(key = lambda x: x[1])
    for job in after_cut:
        RO_2.append(job[0])


  
#    print(RO_1, RO_2)
    return (RO_1, RO_2)



def Constructive_heuristic(RO_1): #Input for the constructive heuristic is a job vector produced in the extended APMX

    for machine in Machinedictionairy:
         Machinedictionairy[machine].Reset_machine
    machine_vector_RO_1 = []
    
    for job in RO_1:
        processingtime = Operationdictionairy[job].processingtime
        required_toolset = Operationdictionairy[job].toolset
        toolset_size = Operationdictionairy[job].toolsetsize
        
        #Allocate job to a machine
        i = 0
        #Check if there exists a machine holding required_toolset, if so set i equal to 1
        for machine in Machinedictionairy:
            for toolset_tuple in Machinedictionairy[machine].toolsets_and_sizes:
                if toolset_tuple[0] == required_toolset: #M_t exists and i is set equal to 1
                    machine_name = machine
#                    print(job)
                    i = 1
                    break
            
        if i == 1: #M_t exists and the job is allocated to M_t
            Machinedictionairy[machine_name].sum_processingtimes += processingtime
            machine_vector_RO_1.append(machine_name)
        else:            
            #M_t does not exist and has to be choosen
            M_c = []
            for machine in Machinedictionairy:
                if (Machinedictionairy[machine].capacity - toolset_size) >= 0:
                    for unit in M_c:
                        if Machinedictionairy[machine].capacity == unit[1]: #there are duplicates, if machine has the lowest index, delete unit from the list and add machine. Otherwise, do nothing
                            index_machine = machine[-1]
                            index_unit = unit[0][-1]
                            if index_machine < index_unit:
                                M_c.remove(unit)
                                M_c.append((machine, Machinedictionairy[machine].capacity, Machinedictionairy[machine].sum_processingtimes))
                    #No duplicates found, append machine to M_c
                    M_c.append((machine, Machinedictionairy[machine].capacity, Machinedictionairy[machine].sum_processingtimes))
            
            #Check if there are machines with sufficient capacity or not
            if len(M_c) > 0:
#                print(job)
#                print(M_c)
                #Sort M_c using processing times
                M_c.sort(key = lambda x: x[2])
                #Select the first machine from the sorted list, this is the machine with the lowest processingtime
                machine = M_c[0][0]
                Machinedictionairy[machine].add_toolset((required_toolset, toolset_size))
                Machinedictionairy[machine].change_capacity(toolset_size)
                Machinedictionairy[machine].sum_processingtimes += processingtime
                machine_vector_RO_1.append(machine)
            else: #No machine with sufficient capacity. Out of all machines, select the one with the smallest processing time
#                print(job)
                smallest_processingtime = 10000 #High initial processing time, so the processing time of the fist machine will always be lower
                for machine in Machinedictionairy:
                    current_processingtime = Machinedictionairy[machine].sum_processingtimes
                    if current_processingtime < smallest_processingtime:
                        smallest_processingtime = current_processingtime
                        machine_name = machine
                Machinedictionairy[machine_name].add_toolset((required_toolset, toolset_size))
                Machinedictionairy[machine_name].change_capacity(toolset_size)
                Machinedictionairy[machine_name].sum_processingtimes += processingtime
                machine_vector_RO_1.append(machine)
    
    Reset_classes()
#    print(machine_vector_RO_1)

    return (machine_vector_RO_1)


    
def uniform_mutation(machine_vector):
    Parent1_machine_vector=machine_vector
    z=0
    for x in Parent1_machine_vector:
        if np.random.random()<= P_u:
            b=np.random.randint(0, len(machinelist))
            Parent1_machine_vector[z]=machinelist[b]
        z=z+1
    return (Parent1_machine_vector)

def swap_mutation(job_vector):

    Parent1_job_vector=job_vector
    c=0
    for x in Parent1_job_vector:
        if np.random.random()<= P_s:
            b=np.random.randint(0,len(job_vector) )
            swapped_value=x
            swapped_with=Parent1_job_vector[b]
            Parent1_job_vector[c]=swapped_with
            Parent1_job_vector[b]=swapped_value
        c=c+1
    return (Parent1_job_vector)
            


def fitness_evaluation(job_vector, machine_vector):
    total_tardiness=0
    Total_toolchanges=0
    for machine in machinelist:
        x=-1
        for job in job_vector:
            x=x+1
            if machine_vector[x]==machine:
                Operation_before_finish=0
                if Operationdictionairy[job].operation>1:
                    for job_before in job_vector:
                        if (Operationdictionairy[job].job==Operationdictionairy[job_before].job) and ((Operationdictionairy[job_before].operation)==(Operationdictionairy[job].operation-1)):
                            Operation_before_finish=Operationdictionairy[job_before].endtime
                if Operationdictionairy[job].tool_switch==1:
                    Total_toolchanges=Total_toolchanges+1
                e_ij=max(Operationdictionairy[job].releasetime, Machinedictionairy[machine].time_finished,Operation_before_finish)+ Operationdictionairy[job].tool_switch +Operationdictionairy[job].processingtime
                Machinedictionairy[machine].add_job((e_ij+Operationdictionairy[job].processingtime+Operationdictionairy[job].tool_switch) )
                Operationdictionairy[job].endtime=(e_ij+Operationdictionairy[job].processingtime+Operationdictionairy[job].tool_switch)
                tardiness=max(0, (Operationdictionairy[job].endtime - Operationdictionairy[job].duedate))
                total_tardiness=total_tardiness+tardiness
                score=Weight_tardiness*total_tardiness+Weight_tool_setup*Total_toolchanges
    Reset_classes()
    return (score)
                
                
            
    
    
def elitism_and_immigration(id_and_scores_list, job_vectors_list_offspring, machine_vectors_list_offspring):
    S_e=int(Y_2*N_p)
    id_and_scores_list.sort(key = lambda x: x[2])
    best_parents_id_and_score = id_and_scores_list[:S_e]
    job_vectors_list_best_parents=[]
    machine_vectors_list_best_parents=[]
    for x in best_parents_id_and_score:
        job_id = x[0]
        job_vectors_list_best_parents.append(job_id)
        machine_id = x[1]
        machine_vectors_list_best_parents.append(machine_id)
        
    ##now randomly choose S_e numbers from set of offsprings
    i = 0
    for b in range(0, S_e):
        z= np.random.randint(0, (len(job_vectors_list_offspring))) #Randomly select offspring chromosome in the form of an index to search in the job and machine vector lists
        best_parent_job = job_vectors_list_best_parents[i]
        x=0
        while job_vectors_list_offspring[z] == best_parent_job:#Change random offspring job vector for a best parent job vector
             z= np.random.randint(0, (len(job_vectors_list_offspring)))
             x=x+1
             if x>40:
                 break
        job_vectors_list_offspring[z]=best_parent_job
        best_parent_machine = machine_vectors_list_best_parents[i]
        machine_vectors_list_offspring[z] = best_parent_machine #Change random offspring machine vector for a best parent machine vector
        i += 1
        
#    #Replace duplicate chromosomes by randomly initiated new ones
#    #Create new random chromosomes
#    output1 = Random_initialization(Jobb_vector_init, machinelist) ## returns a list with all parents machine vectors
#    random_job_vectors = output1[0] ## note moet nog even N_p-1 doen want practicioner wordt nog toegevoegd. Gedaan in de random_init functie
#    random_machine_vectors = output1[1]
#
#    new_job_vectors_list_offspring = []
#    new_machine_vectors_list_offspring = []
#    w = 0
#    for x in range(0, (len(job_vectors_list_offspring) - 1)):
#        job_vector = job_vectors_list_offspring[x]
#        machine_vector = machine_vectors_list_offspring[x]
#        for y in job_vectors_list_offspring[x:]: #range(x, (len(job_vectors_list_offspring) - x)):
#            if job_vector == y: #Duplicate found
#                new_job_vectors_list_offspring.append(job_vector) #add the duplicate job vector once to the new list of job vectors
#                new_machine_vectors_list_offspring.append(machine_vector) #add the duplicate machine vector once to the new list of machine vectors
#                random_job_vector = random_job_vectors[w]
#                new_job_vectors_list_offspring.append(random_job_vector) #add random initiated job vector to new list of job vectors
#                random_machine_vector = random_machine_vectors[w]
#                new_machine_vectors_list_offspring.append(random_machine_vector) #add random initiated machine vector to new list of machine vectors
#                w += 1
#            else: #No duplicate found
#                new_job_vectors_list_offspring.append(job_vector)
#                new_machine_vectors_list_offspring.append(machine_vector)
#    
#    job_vectors_list_offspring = new_job_vectors_list_offspring
#    machine_vectors_list_offspring = new_machine_vectors_list_offspring
    
    return(job_vectors_list_offspring, machine_vectors_list_offspring)
        
    
      
    
def Proposed_tool_replacement(job_vector, machine_vector):
    tool_changes=0

    for machine in machinelist:
        operations_to_be_done = job_vector
        machine_to_be_done = machine_vector
        list_T=[]
        while len(operations_to_be_done)>1:
            Operations_done=0
            T_ij=0
            for x in range (0, len(operations_to_be_done)):
                Operations_done=Operations_done+1
                Operation=operations_to_be_done[x]
                if machine==machine_to_be_done[x]:
                   if (int(Machinedictionairy[machine].capacity)>=int(Operationdictionairy[Operation].toolsetsize)) and  not int(Operationdictionairy[Operation].toolset) in list_T:
                       Machinedictionairy[machine].change_capacity(Operationdictionairy[Operation].toolsetsize)
                       Machinedictionairy[machine].add_toolset((Operationdictionairy[Operation].toolset, Operationdictionairy[Operation].toolsetsize ))
                       list_T.append(Operationdictionairy[Operation].toolset)
                   elif (Machinedictionairy[machine].capacity<Operationdictionairy[Operation].toolsetsize) and T_ij==0 and  not Operationdictionairy[Operation].toolset in list_T:
                       T_ij=(Operationdictionairy[Operation].toolset, Operationdictionairy[Operation].toolsetsize )
                       Operationdictionairy[Operation].tool_switch=1
                       break
            list_T=[]  

            for toolset in Machinedictionairy[machine].toolsets_and_sizes:
                list_T.append(toolset[0])
            list_T=list(set(list_T))
            operations_to_be_done=operations_to_be_done[Operations_done:]
            machine_to_be_done=machine_to_be_done[Operations_done:]
            if len(operations_to_be_done)>0:
                list_TS=[]
                input_lp_v1=[]
                for z in range (0, len(operations_to_be_done)):
                    Operation=operations_to_be_done[z]
                    if machine==machine_to_be_done[z]:
                        if Operationdictionairy[Operation].toolset in list_T:
                            list_TS.append(Operationdictionairy[Operation].toolset)
                list_TS=list(set(list_TS))
                c=0
                list_do_not_count_twice=[]
                for z in range (0, len(operations_to_be_done)):
                    Operation=operations_to_be_done[z]
                    if machine==machine_to_be_done[z]:
                        if Operationdictionairy[Operation].toolset in list_TS:
                            if not Operationdictionairy[Operation].toolset in list_do_not_count_twice:
                                input_lp_v1.append((Operationdictionairy[Operation].toolset, (len(list_TS)-c)))
                                list_do_not_count_twice.append(Operationdictionairy[Operation].toolset)
                                c=c+1
                for x in list_T:
                    if not x in list_TS:
                        input_lp_v1.append((x, 0))
                        
                input_lp_v2=[]
                for y in input_lp_v1:
                    for q in Machinedictionairy[machine].toolsets_and_sizes:
                        if y[0]==q[0]:
                            input_lp_v2.append((y[0],y[1],q[1]))
                binaryvariables=[]
                scores=[]
                sizes=[]
                for x in input_lp_v2:
                   binaryvariables.append(x[0])
                   scores.append(x[1])
                   sizes.append(x[2])
                lp_model=gp.Model("MIP")
                lp_model.Params.LogToConsole = 0
                for x in binaryvariables:
                    vname=str(x)
                    x=lp_model.addVar(vtype=GRB.BINARY, name=vname)
                lp_model.update()
                objective=sum(lp_model.getVars()[x]*scores[x] for x in range (0, len(scores)-1))
                lp_model.setObjective(objective, GRB.MINIMIZE)
                constraint=(sum((lp_model.getVars()[x]*sizes[x] for x in range (0, len(sizes))))+Machinedictionairy[machine].capacity)>= T_ij[1]
                lp_model.update()
                lp_model.addConstr(constraint)
                lp_model.update()
                lp_model.optimize()
#                print ('The toolset and sizes of ' + machine + ' at the start of this period were :' , Machinedictionairy[machine].toolsets_and_sizes)
                deleted_toolset_list=[]
                deleted_toolsetsize=0
                for x in range(0,len(scores)):
                    if lp_model.getAttr(GRB.Attr.X, lp_model.getVars())[x]>0:
                        deleted_toolset=input_lp_v2[x][0]
                        deleted_toolsetsize=deleted_toolsetsize+input_lp_v2[x][2]
                        deleted_toolset_list.append(deleted_toolset)
                        Machinedictionairy[machine].toolsets_and_sizes=[x for x in (Machinedictionairy[machine].toolsets_and_sizes) if x[0]!=deleted_toolset]
                Machinedictionairy[machine].add_toolset(T_ij)
                new_capacity=int(capacity_per_machine)
                for x in Machinedictionairy[machine].toolsets_and_sizes:
                    new_capacity=new_capacity- x[1]
                Machinedictionairy[machine].capacity=new_capacity
                lp_model.reset()
#                print('toolsets are deleted: ', deleted_toolset_list)
#                print('toolset is added: ' , T_ij[0])
#                print ('The toolset and sizes of ' + machine + ' at the end of this period were :' , Machinedictionairy[machine].toolsets_and_sizes)
                tool_changes=tool_changes+1          
#    print(tool_changes)
    #Changes binary values within classes so no return needed
    #Do not reset classes, because the binary information in the classes is used in the fitness evaluation

             
                    
def Genetic_Algorithm(list_PH_job_vectors, list_PH_machine_vectors, Jobb_vector_init, machinelist, N_p):
    start = time.clock()
    best = False #Line 1 pseudocode
    k=1 ##generatie 1
    q = 0 #Needed to initialize q in line 7 pseudocode
    output1=Random_initialization(Jobb_vector_init, machinelist) ## returns a list with all parents machine vectors
    list_job_vectors_parents=output1[0] ## note moet nog even N_p-1 doen want practicioner wordt nog toegevoegd. Gedaan in de random_init functie
    list_machine_vectors_parents=output1[1]
    ## now the operations are good ordered so Operation1_1 will be before operation1_2
    b=0
    for x in list_job_vectors_parents:
        x_new=job_vector_re_operation_sorter(x) ## note: deze functie werkt nog alleen voor 2 operaties dus nog aanpassen
        list_job_vectors_parents[b]=x_new
        b=b+1
    if k==1:
        for x in range(0, len(list_PH_job_vectors)):
            list_job_vectors_parents.append(list_PH_job_vectors[x]) ## hier practcioner toevoegen
            list_machine_vectors_parents.append(list_PH_machine_vectors[x])
        
    
    fitness_values_corresponding_to_parents=[]

    for x in range(0,len(list_job_vectors_parents)):
        job_vector=list_job_vectors_parents[x]
#        print(job_vector)
        machine_vector=list_machine_vectors_parents[x]
#        print(machine_vector)
        Proposed_tool_replacement(job_vector, machine_vector) ## here all toolswitches are calculated and when a toolswitch is neded for a operation the binary value is set to 1
        output2=fitness_evaluation(job_vector, machine_vector) ## here the fitness value is calculated 
        fitness_values_corresponding_to_parents.append(output2) ## here the values are stored in the list
#        Reset_classes() ## reset classes so that all the classes start again as in the initialized state
#        Heb de Reset_classes in de fitness evaluation functie gezet, zodat dit niet vergeten kan worden om te doen

#    f_best=10000000000000 # start very high so that the begin value is always higher than the f_best    
#    f_k = min(fitness_values_corresponding_to_parents) #f_k is the highest of the generation
#    if f_k<f_best:
#        f_best=f_k ## f_best is the highest of all generations
#    index_f_k = fitness_values_corresponding_to_parents.index(min(fitness_values_corresponding_to_parents)) #Thought we needed the index of the lowest value but we need the actual value, left it in in case we need it later on to retrieve the corresponding vectors
    
    #Commented the code above, because we only have to initialize f_best in line 5 and not yet f_k
    f_best = min(fitness_values_corresponding_to_parents) #f_k is the highest of the generation

    
    #keep track of expired time using time
    #keep track of number of times f_best did not change using not_improved_f_counter
    not_improved_f_counter = 0
    while (time.clock()-start) < maxtime and not_improved_f_counter < G_c: #maxtime and G_c follow from chapter 7 and are the termination criteria for the matheuristic
        print(timeit.default_timer())
        if best == True or q < B:  ## twijfel of dit or of and moet zijn eigenlijk, als ik de tekst lees, maar in pseudocode staat or toch?
            list_C_k_job_vectors=[]
            list_C_k_machine_vectors=[]            
            number_offspring_created = 0
            while number_offspring_created < N_p: #Loop to create N_p offspring chromosomes
                s_T=int(Y_1*N_p) ## here the population of the tournament is calculated
                index_parents=[]
                for b in range (0, 2): ## 2 parents need to be selected
                    tournament_selection_indexs=[] ## here the indexes are sorted
                    tournament_selection_fitness_values=[] ## here the fitness values are sorted
                    for x in range (0, s_T): ## here they are selected
                        a=np.random.randint(0, len(fitness_values_corresponding_to_parents))
                        while a in tournament_selection_indexs:    
                           a=np.random.randint(0, len(fitness_values_corresponding_to_parents))
                        tournament_selection_indexs.append(a)
                        tournament_selection_fitness_values.append(fitness_values_corresponding_to_parents[a])
                    index_lowest_fitness=tournament_selection_fitness_values.index(min(tournament_selection_fitness_values))
                    index_parent=tournament_selection_indexs[index_lowest_fitness]
                    index_parents.append(index_parent)
                job_vector_parent1=list_job_vectors_parents[index_parents[0]] ## take the index out of the list of total parents
                job_vector_parent2=list_job_vectors_parents[index_parents[1]]
                machine_vector_parent1=list_machine_vectors_parents[index_parents[0]]
                machine_vector_parent2=list_machine_vectors_parents[index_parents[1]]
                CK_job_vector=Extended_APMX_crossover(job_vector_parent1, job_vector_parent2) ## put the job vectors in the extended APMX
                CK_job_vector1=CK_job_vector[0]
                CK_job_vector2=CK_job_vector[1]
                list_C_k_job_vectors.append(CK_job_vector1) ## add job vectors the the list of C_k
                list_C_k_job_vectors.append(CK_job_vector2)
                machine_vector_CK_1=Constructive_heuristic(CK_job_vector1) ## add machine to machine vector C_k
                machine_vector_CK_2=Constructive_heuristic(CK_job_vector2)
                list_C_k_machine_vectors.append(machine_vector_CK_1)
                list_C_k_machine_vectors.append( machine_vector_CK_2)
                number_offspring_created += 2
        else:
            list_C_k_job_vectors=[]
            list_C_k_machine_vectors=[]   
            number_offspring_created = 0
            while number_offspring_created < N_p: #Loop to create N_p offspring chromosomes
                s_T=int(Y_1*N_p) ## here the population of the tournament is calculated
                index_parents=[]
                for b in range (0, 2): ## 2 parents need to be selected
                    tournament_selection_indexs=[] ## here the indexes are sorted
                    tournament_selection_fitness_values=[] ## here the fitness values are sorted
                    for x in range (0, s_T): ## here they are selected
                        a=np.random.randint(0, len(fitness_values_corresponding_to_parents))
                        while a in tournament_selection_indexs:    
                           a=np.random.randint(0, len(fitness_values_corresponding_to_parents))
                        tournament_selection_indexs.append(a)
                        tournament_selection_fitness_values.append(fitness_values_corresponding_to_parents[a])
                    index_lowest_fitness=tournament_selection_fitness_values.index(min(tournament_selection_fitness_values))
                    index_parent=tournament_selection_indexs[index_lowest_fitness]
                    index_parents.append(index_parent)
                print('index parents: ' + str(index_parents))
                print('list job vector parents: ' + str(len(list_job_vectors_parents)))
                job_vector_parent1=list_job_vectors_parents[index_parents[0]] ## !!!take the index out of the list of total parents
                job_vector_parent2=list_job_vectors_parents[index_parents[1]]
                machine_vector_parent1=list_machine_vectors_parents[index_parents[0]]
                machine_vector_parent2=list_machine_vectors_parents[index_parents[1]]
                CK_job_vector=APMX_crossover(job_vector_parent1, job_vector_parent2) ## put the job vectors in the extended APMX
                CK_job_vector1=CK_job_vector[0]
                CK_job_vector2=CK_job_vector[1]
                list_C_k_job_vectors.append(CK_job_vector1) ## add job vectors the the list of C_k
                list_C_k_job_vectors.append(CK_job_vector2)
                CK_machine_vector = two_point_crossover(machine_vector_parent1, machine_vector_parent2)
                machine_vector_CK_1 = CK_machine_vector[0] ## add machine to machine vector C_k
                machine_vector_CK_2 = CK_machine_vector[1]
                list_C_k_machine_vectors.append(machine_vector_CK_1)
                list_C_k_machine_vectors.append( machine_vector_CK_2)
                number_offspring_created += 2
        #Use uniform mutation on job vector C_k and swap mutation on machine vector C_k
        list_C_k_mutated_job_vectors = []
        list_C_k_mutated_machine_vectors = []
        for machine_vector in list_C_k_machine_vectors:
            mutated_machine_vector = uniform_mutation(machine_vector)
            list_C_k_mutated_machine_vectors.append(mutated_machine_vector)
        for job_vector in list_C_k_job_vectors:
            mutated_job_vector = swap_mutation(job_vector)
            reordered_job_vector = job_vector_re_operation_sorter(mutated_job_vector) #Why is reordering needed?
            list_C_k_mutated_job_vectors.append(reordered_job_vector)
        #Compute tool replacements for C_k_mutated and evaluate C_k_mutated
        fitness_values_corresponding_to_C_k_mutated = []
        for x in range(0, len(list_C_k_mutated_job_vectors)):
            job_vector = list_C_k_mutated_job_vectors[x]
            machine_vector = list_C_k_mutated_machine_vectors[x]
            Proposed_tool_replacement(job_vector, machine_vector) ## here all toolswitches are calculated and when a toolswitch is neded for a operation the binary value is set to 1
            output2 = fitness_evaluation(job_vector, machine_vector) ## here the fitness value is calculated 
            fitness_values_corresponding_to_C_k_mutated.append(output2) ## here the values are stored in the list
        #Create P_(k + 1) from P_k and C_k_mutated by elitism selection and immigration
        #Make id list with tuples containing (job id, machine id, fitness value) for all parents
        id_and_scores_list = [] ## ik denk dat ik dit deel onnodig heb gemaakt door mijn aanpassingen.
        for x in range(0, len(list_job_vectors_parents)):
            job_vector = list_job_vectors_parents[x]
            machine_vector = list_machine_vectors_parents[x]
            fitness_value = fitness_values_corresponding_to_parents[x]
            id_and_scores_list.append((job_vector, machine_vector, fitness_value))
        
        P_k_new = elitism_and_immigration(id_and_scores_list, list_C_k_mutated_job_vectors, list_C_k_mutated_machine_vectors)
        list_job_vectors_parents = P_k_new[0]
        list_machine_vectors_parents = P_k_new[1]
        #Determine f_k
        fitness_values_corresponding_to_k_new = []
        for x in range(0, len(P_k_new[0])):
            job_vector = P_k_new[0][x]
            machine_vector = P_k_new[1][x]
            Proposed_tool_replacement(job_vector, machine_vector) ## here all toolswitches are calculated and when a toolswitch is neded for a operation the binary value is set to 1
            output2 = fitness_evaluation(job_vector, machine_vector) ## here the fitness value is calculated 
            fitness_values_corresponding_to_k_new.append(output2)
        
        f_k=min(fitness_values_corresponding_to_k_new)
        id_and_scores_list = [] ## ik denk dat ik dit deel onnodig heb gemaakt door mijn aanpassingen.
        
        
        fitness_values_corresponding_to_parents=[]

        for x in range(0,len(list_job_vectors_parents)):
            job_vector=list_job_vectors_parents[x]
#        print(job_vector)
            machine_vector=list_machine_vectors_parents[x]
#        print(machine_vector)
            Proposed_tool_replacement(job_vector, machine_vector) ## here all toolswitches are calculated and when a toolswitch is neded for a operation the binary value is set to 1
            output2=fitness_evaluation(job_vector, machine_vector) ## here the fitness value is calculated 
            fitness_values_corresponding_to_parents.append(output2) ## here the values are stored in the list
        


        print(f_k, f_best)
        if f_k < f_best:
            not_improved_f_counter=0
            f_best = f_k
            best = True
            q = 1
        else:
            best = False
            q += 1
            not_improved_f_counter += 1
            
    stop = time.clock() 
    print('Computaton_time : ' + str(stop-start))
    print('Objective_value : ' + str(f_best))
    return(f_best)

          
                    
Genetic_Algorithm(list_PH_job_vectors, list_PH_machine_vectors, Jobb_vector_init, machinelist, N_p)
                
                
                    
            
                


        
        
        
        
            

        
                    
        
                        
                        
                    
                    
                    
                    
                    
                
                
            
            
            
            

            
            
        
    

    
# Fitness evaluation: