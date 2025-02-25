satellite-simuation
===================

## Introduction
  This program plot the trajectory of satellites which has TLE file in online sites. The TLE file has various data such as the relative position and orbit information of the satellite. The presence of a TLE file allows you to know the location of the satellite at a specific time. Therefore, based on this series of data, we intend to produce a program that visualizes satellites. The program was produced using Pyvista. This program was implemented using gRPC as an independent program. Therefore, it is easily available in other external programs. This simulation use ECEF cordinates. x axis is where lattitude and longitude is 0, z axis is the rotation axis of earth, and y-axis is cross product of x and z axis.

## How To See
  Red means x-axis, green means y-axis and blue means z-axis. white trajectory means when the satellite is exposed to sunlight, gray trajctory menas not exposed. The yellow thick line means the communication avaliable range.
  
## Usage
  This application is implemented with gRPC, So you can easily access this program. This is earth.proto which made earth_pb2.py. To plot the trajectory, You need to send three basic information to server. Those are SatTrajectoryInfo, SatAttitudeInfo, GroundStationInfo. SatTrajectoryInfo contains the time and the satellite SN. SatAttitudeInfo contains quaternion. Lastly, GroundStationInfo contains the degree which set the communication range and location. We need to send this information to server, but the order exists. SatTrajectoryInfo must be sent before GroundStationInfo. Because It calculates commnuncation available range using the SatTrajectoryInfo. If you send whole information to server, and you can call StartDraw to visualize the models.

```bash
syntax = "proto3";

service Satrac{
  rpc SendTrajectoryInfo (SatTrajectoryInfo) returns (Response_code) {}
  rpc SendAttitudeInfo (SatAttitudeInfo) returns (Response_code) {}
  rpc SendGroundStationInfo (GroundStationInfo) returns (Response_code) {}
  rpc SendSatVecInfo (stream SatVectorInfo) returns (Response_code) {}
  rpc ChangeUrl (Url) returns (Response_code) {}
  rpc StartDraw (Empty) returns (Response_code) {}
}

message SatTrajectoryInfo {
  SatDate t = 1;
  int32 sat_num = 2;
}
message GroundStationInfo {
  float x = 1;
  float y = 2;
  float z = 3;
  float degree = 4;
}
message Url {
  string url = 1;
}
message SatDate {
  int32 y = 1;
  int32 mo = 2;
  int32 d = 3;
  int32 h = 4;
  int32 m = 5;
  int32 s = 6;
}
message TLEInfo {
  string line1 = 1;
  string line2 = 2;
  string line3 = 3;
}
message SatAttitudeInfo {
  float q1 = 1;
  float q2 = 2;
  float q3 = 3;
  float q4 = 4;
}
message SatVectorInfo {
  float u = 1;
  float v = 2;
  float w = 3;
}
message Response_code {
  string message = 1;
}

message Empty {}
``` 
  The Below code is example of client.py which start program with some inital condition. It sends data in following order: Trajectory, GroundSatation, Attitude and etc Vectors. It connects with localhost:50051. You can see how to send each parameter using gRPC. 
```python
from __future__ import print_function

import logging
import math
 
import grpc
import earth_pb2
import earth_pb2_grpc


def vec_stream_generator(vec):
    for ele in vec:
        yield earth_pb2.SatVectorInfo(u = ele[0], v = ele[1], w = ele[2])


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = earth_pb2_grpc.SatracStub(channel)

        t_in = earth_pb2.SatDate(y=2022, mo=7, d=18, h=15, m=15, s=0)
        sat_num_in = 52935

        vec_stream = vec_stream_generator([(math.sin(1), 0, math.cos(1)),(0,math.sin(1),math.cos(1))])

        
        response = stub.SendTrajectoryInfo(earth_pb2.SatTrajectoryInfo(t = t_in, sat_num = sat_num_in))
        print(response)
        response = stub.SendGroundStationInfo(earth_pb2.GroundStationInfo(x = 6378.1, y = 0, z = 0, degree = 0))
        print(response)
        response = stub.SendAttitudeInfo(earth_pb2.SatAttitudeInfo(q1=(1/2)**(1/2), q2=0, q3=0, q4=(1/2)**(1/2)))
        print(response)
        response = stub.SendSatVecInfo(vec_stream)
        print(response)
        response = stub.StartDraw(earth_pb2.Empty())
        print(response)
        


if __name__ == '__main__':
    logging.basicConfig()
    run()
```