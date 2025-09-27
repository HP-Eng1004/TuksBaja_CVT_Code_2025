# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:12:48 2017

@author: User
"""
import numpy as np
from collections import namedtuple
def readV3( filename ):
    
    fid = open(filename, "rb")
    
    Version = np.fromfile(fid, dtype=np.uint16,count=1)
    print("Version",Version[0])
    if(Version[0] == 3):
        ch = [1];
        HeaderRead = 0
        temp=[0, 0, 0]
        stop = 0;
        Data =[]
        while (ch[0] != "") & (stop == 0):
            ch = np.fromfile(fid, dtype=np.uint8,count=1)
            if(ch[0]==""):
                break;
            if((ch[0]==int('0xA0',16)) & (HeaderRead==0)):
                temp[0] = ch[0];
            elif((ch[0]==int('0xA1',16)) & (temp[0] == int('0xA0',16))):    
                temp[1] = ch[0];
            elif((ch[0]==int('0xA2',16)) & (temp[0] == int('0xA0',16)) & (temp[1] == int('0xA1',16))):   
                temp[2] = ch[0];
            elif((ch[0]==int('0xA3',16)) & (temp[0] == int('0xA0',16)) & (temp[1] == int('0xA1',16)) & (temp[2] == int('0xA2',16))):    
                
                print('start Header')
                HeaderRead = 1
                SFRead = 0
                temp2=[0, 0, 0]
                temp=[0, 0, 0]
                temp3=[0, 0, 0]
                Nchannelset = 0
                Channel = namedtuple('Channel', ['ChannelName','ChannelNumber','Data','time'])
                ChannelFound = 0;
                NChannel = -1;
                while stop != 1:
                    ch = np.fromfile(fid, dtype=np.uint8,count=1)
                    if(ch[0]==""):
                        break;
                        

                    if((ch[0]==int('0xB0',16)) & (HeaderRead==1)):
                        temp[0] = ch[0];
                    elif((ch[0]==int('0xB1',16)) & (temp[0] == int('0xB0',16))):    
                        temp[1] = ch[0];
                    elif((ch[0]==int('0xB2',16)) & (temp[0] == int('0xB0',16)) & (temp[1] == int('0xB1',16))):   
                        temp[2] = ch[0];
                    elif((ch[0]==int('0xB3',16)) & (temp[0] == int('0xB0',16)) & (temp[1] == int('0xB1',16)) & (temp[2] == int('0xB2',16))):
                        print('stop Header')
                    else:
                        temp=[0, 0, 0]
                        
                    if(SFRead == 0):
                        if((ch[0]==int('0xC0',16)) ):
                            temp2[0] = ch[0];
                        elif((ch[0]==int('0xC1',16)) & (temp2[0] == int('0xC0',16))):  
                            SFtemp = np.fromfile(fid, dtype=np.float32,count=1)[0]
                            temp2[1] = ch[0];
                        elif((temp2[1]==int('0xC1',16)) & (temp2[0] == int('0xC0',16)) & (ch[0] == int('0xD0',16))):
                            temp2[2] = ch[0];
                        elif((temp2[1]==int('0xC1',16)) & (temp2[0] == int('0xC0',16)) & (temp2[2] == int('0xD0',16))& (ch[0] == int('0xD1',16))):
                            SF = SFtemp
                            SFRead = 1
                            print('SF = ',SF)
                            temp2=[0, 0, 0]
                            
                            
                    if(Nchannelset == 0):
                        if((ch[0]==int('0xC2',16)) ):
                            temp2[0] = ch[0];
                        elif((ch[0]==int('0xC3',16)) & (temp2[0] == int('0xC2',16))):  
                            NChanneltemp = np.fromfile(fid, dtype=np.uint8,count=1)[0]
                            temp2[1] = ch[0];
                        elif((temp2[1]==int('0xC3',16)) & (temp2[0] == int('0xC2',16)) & (ch[0] == int('0xD2',16))):
                            temp2[2] = ch[0];
                        elif((temp2[1]==int('0xC3',16)) & (temp2[0] == int('0xC2',16)) & (temp2[2] == int('0xD2',16))& (ch[0] == int('0xD3',16))):
                            NChannel = NChanneltemp
                            Nchannelset = 1
                            print('NChannel = ',NChannel)
                            temp2=[0, 0, 0]            
                    
                
                    if(Nchannelset == 1):
                        if((ch[0]==int('0xC4',16)) ):
                            temp2[0] = ch[0];
                        elif((ch[0]==int('0xC5',16)) & (temp2[0] == int('0xC4',16))):  
                            ChannelNumTemp = np.fromfile(fid, dtype=np.uint8,count=1)[0]
                            ChannelNameLenTemp = np.fromfile(fid, dtype=np.uint8,count=1)[0]
                            NameTemp = fid.read(ChannelNameLenTemp)  
                            temp2[1] = ch[0];
                        elif((temp2[1]==int('0xC5',16)) & (temp2[0] == int('0xC4',16)) & (ch[0] == int('0xD5',16))):
                            temp2[2] = ch[0];
                        elif((temp2[1]==int('0xC5',16)) & (temp2[0] == int('0xC4',16)) & (temp2[2] == int('0xD5',16))& (ch[0] == int('0xD6',16))):
                            ChannelNum = ChannelNumTemp
                            Name = NameTemp
                            print('Name = ',Name)
                            temp2=[0, 0, 0]
                            Data.append(Channel(Name.decode("utf-8") , ChannelNum, np.array([]),np.array([]) ) )
                            ChannelFound+=1
                            k = 0;
                
                    if(ChannelFound == NChannel ):
                        if((ch[0]==int('0xE0',16)) & (HeaderRead==1)):
                            temp3[0] = ch[0];
                        elif((ch[0]==int('0xE1',16)) & (temp3[0] == int('0xE0',16))):    
                            temp3[1] = ch[0];
                        elif((ch[0]==int('0xE2',16)) & (temp3[0] == int('0xE0',16)) & (temp3[1] == int('0xE1',16))):   
                            temp3[2] = ch[0];
                        elif((ch[0]==int('0xE3',16)) & (temp3[0] == int('0xE0',16)) & (temp3[1] == int('0xE1',16)) & (temp3[2] == int('0xE2',16))):
                            print('start Data')
                            DataStart = fid.tell()
                            fid.seek(-4,2) # move the cursor to the end of the file
                            size = fid.tell()
                            NumPoints = (int)((size-DataStart)/NChannel/2);
                            time = np.array(range(0,NumPoints))/SF
                            fp = np.memmap(filename, offset=DataStart, dtype='int16', mode='r', shape=(NumPoints,NChannel))
                            #fp = np.memmap(filename, offset=DataStart, dtype='int16', mode='r')
                            arr = np.array(fp[:]);
                            arr = arr.astype('float');
                            arr = (arr*10)/32768
                            for i in range(0,NChannel):
                                Data[i] = Data[i]._replace(Data=arr[:,i])
                                Data[i] = Data[i]._replace(time=time)
                            stop = 1  
                        else:
                            temp3=[0, 0, 0]        
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
            else:
                temp =[0, 0, 0]
    return Data