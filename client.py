from __future__ import print_function

import logging
import math
 
import grpc
import earth_pb2
import earth_pb2_grpc


def vec_stream_generator(vec):
    for ele in vec:
        yield earth_pb2.SatVectorInfo(u = ele[0], v = ele[1], w = ele[2])


def run(port):
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
    run('')