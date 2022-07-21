from concurrent import futures
from locale import normalize
import logging

from skyfield.api import load, wgs84

import grpc
import earth_pb2
import earth_pb2_grpc

import pyvista as pv
import math
import numpy as np


class Satrac(earth_pb2_grpc.SatracServicer):
    def __init__(self):
        self.station_url = 'https://celestrak.org/NORAD/elements/gp.php?INTDES=2022-072'
        self.view_degree = 0
        self.earth_radius = 6378.1
        self.sat_vec = []
        self.eph = load('de421.bsp')
        
        self.Trajectoryflag = False
        self.GroundStationinfoflag = False
        self.Attitudeinfoflag = False

    def StartDraw(self, request, context):
        if self.Trajectoryflag == False:
            return earth_pb2.Response_code(message = 'The Error Occured : %s' % ('Trajectoryflag is diabled'))
        
        if self.GroundStationinfoflag == False:
            return earth_pb2.Response_code(message = 'The Error Occured : %s' % ('GSinfoflag is diabled'))

        if self.Attitudeinfoflag == False:
            return earth_pb2.Response_code(message = 'The Error Occured : %s' % ('Attitudeinfoflag is diabled'))


        try:
            self.start_program()
        except Exception as e:
            return earth_pb2.Response_code(message = 'The Error Occured while starting program : %s' % e)

        return earth_pb2.Response_code(message = 'Success : StartDraw')

    def ChangeUrl(self, request, context):
        self.station_url = request.url
        
        return earth_pb2.Response_code(message = 'Success : ChangeUrl')
    
    def SendGroundStationInfo(self, request, context):
        self.gs_location = np.array([request.x,request.y,request.z])
        self.view_degree = request.degree

        if self.Trajectoryflag == False:
            return earth_pb2.Response_code(message = 'The Error Occured : %s' % ('Satinfoflag is diabled'))

        lat, lon = math.asin(normalize(self.gs_location[2]/self.earth_radius)), math.atan2(self.gs_location[1], self.gs_location[0])
        
        crisp = wgs84.latlon(lat, lon)
        t_deg, events_deg = self.satellite.find_events(crisp, self.now_time, self.end_time, altitude_degrees=self.view_degree)

        two_time_set = []

        taos = self.now_time
        aos_flag = False
        for ti, event in zip(t_deg, events_deg):
            if event == 0:
                taos = ti

            elif event == 2:
                two_time_set.append([taos, ti])

        if aos_flag == True:
            two_time_set.append([taos, self.end_time])

        time_gap = 1/(24*60*60)
        self.sat_trac_communication = []

        for start_utc, end_utc in two_time_set:
            li = []
            while end_utc - start_utc >= 0:
                start_utc += time_gap

                li.append(lat_lon_at(self.satellite, start_utc, self.earth_radius))

            self.sat_trac_communication.append(np.array(li))

        self.GroundStationinfoflag = True


        return earth_pb2.Response_code(message = 'Success : SendGroundStationInfo')
    
    def SendTrajectoryInfo(self, request, context):
        ts = load.timescale()
        self.now_time = ts.utc(request.t.y, request.t.mo, request.t.d, request.t.h, request.t.m, request.t.s)
        self.end_time = ts.utc(request.t.y, request.t.mo, request.t.d, request.t.h, request.t.m, request.t.s + 10000)
        self.time_gap = ts.utc(request.t.y, request.t.mo, request.t.d, request.t.h, request.t.m, range(request.t.s,request.t.s+10000,1))
        sat_num = request.sat_num

        try:
            satellites = load.tle_file(self.station_url, reload=True)
            self.satellite = {sat.model.satnum: sat for sat in satellites}[sat_num]

        except Exception as e :
            return earth_pb2.Response_code(message = 'The Error Occured : %s' % e)

        self.sat_trac = [[],[]]
        sat_trac_bright = []
        sat_trac_dark = []
        key = True

        for ti in self.time_gap:
            loc = lat_lon_at(self.satellite, ti, self.earth_radius)
            sunlit = self.satellite.at(ti).is_sunlit(self.eph)
        
            if sunlit :
                if key == False:
                    if len(sat_trac_dark) : 
                        sat_trac_dark.append(loc)
                        self.sat_trac[1].append(np.array(sat_trac_dark))
                        sat_trac_dark = []
                    
                    key = True
                sat_trac_bright.append(loc)
                
            else:
                if key == True:
                    if len(sat_trac_bright) :
                        sat_trac_bright.append(loc)
                        self.sat_trac[0].append(np.array(sat_trac_bright))
                        sat_trac_bright = []

                    key = False
                sat_trac_dark.append(loc)
                
        if key == True:
            self.sat_trac[0].append(np.array(sat_trac_bright))
        else :
            self.sat_trac[1].append(np.array(sat_trac_dark))


        self.sat_location = lat_lon_at(self.satellite, self.now_time, self.earth_radius)

        self.Trajectoryflag = True

        return earth_pb2.Response_code(message = 'Success : SendTrajectoryInfo')

    def SendAttitudeInfo(self,request,context):
        q1, q2, q3, q4 = request.q1, request.q2, request.q3, request.q4
        
        self.sat_dcm = np.array([[(q4**2 + q1**2 -q2**2-q3**2), 2*(q1*q2+q3*q4), 2*(q1*q3-q2*q4)],
                                 [2*(q1*q2-q3*q4), (q4**2-q1**2 + q2**2 - q3**2), 2*(q2*q3 + q1*q4)],
                                 [2*(q1*q3+q2*q4), 2*(q2*q3 - q1*q4), (q4**2 - q1**2 - q2**2 + q3**2)]])
        
        self.Attitudeinfoflag = True
        return earth_pb2.Response_code(message = 'Success : SendAttitudeInfo')

    def SendSatVecInfo(self,request_iterator,context):
        
        for request in request_iterator:
            self.sat_vec.append(np.array([request.u, request.v, request.w]).T)

        return earth_pb2.Response_code(message = 'Success : SendOtherVecInfo')

    def start_program(self):

        sphere = pv.Sphere(radius=self.earth_radius, theta_resolution=240, phi_resolution=240, start_theta=270.0001, end_theta=270)
        sphere.active_t_coords = np.zeros((sphere.points.shape[0], 2))

        for i in range(sphere.points.shape[0]):
            x, y, z = sphere.points[i, 0]/self.earth_radius, sphere.points[i, 1]/self.earth_radius, sphere.points[i, 2]/self.earth_radius
            x, y, z = normalize(x), normalize(y), normalize(z)
            
            sphere.active_t_coords[i] = [0.5 + math.atan2(-x, y)/(2 * math.pi), 0.5 + math.asin(z)/math.pi]
        
        sphere.rotate_z(-90)

        pl = pv.Plotter()
        pl.add_background_image('starry-night-sky-fit.jpg', scale=1.001)
        pl.add_mesh(sphere, texture=pv.read_texture("earth.jpg"), smooth_shading=False)

        for ele in self.sat_trac[0]:
            pl.add_lines(ele, width = 0.5, color = 'white')
        for ele in self.sat_trac[1]:
            pl.add_lines(ele, width = 0.5, color = 'gray')
        for each_trac in self.sat_trac_communication:
            pl.add_lines(each_trac, width = 5, color = 'yellow')
        
        color = ['red','blue','green','yellow','purple','orange', 'white', 'skyblue']

        reader = pv.get_reader('./library/satellite.stl')
        sat_mesh = reader.read()
        sat_scale = 30

        reader = pv.get_reader('./library/antenna.stl')
        gs_mesh = reader.read()
        gs_scale = 2
        gs_dcm = rotation_matrix_from_vectors(np.array([0,1,0]), np.array(self.gs_location))    

        # 0, 1, 0

        print(self.gs_location.shape)

        for i in range(gs_mesh.points.shape[0]):
            gs_mesh.points[i] = (gs_dcm @ (gs_mesh.points[i].T)).T * gs_scale + self.gs_location

        for i in range(sat_mesh.points.shape[0]):
            sat_mesh.points[i] = ((self.sat_dcm @ (sat_mesh.points[i].T)).T)*sat_scale + self.sat_location
    
        pl.add_mesh(gs_mesh)
        pl.add_mesh(sat_mesh)


        basis = np.array([[500,0,0],
                        [0,500,0],
                        [0,0,500]])
        sat_att = (self.sat_dcm @ basis).T

        for i in range(3):
            pl.add_lines(np.array([self.sat_location + sat_att[i],self.sat_location]), color = color[i], width = 1.5)

        for i in range(len(self.sat_vec)):
            pl.add_lines(np.array([self.sat_location , self.sat_location + 400*(self.sat_dcm@((self.sat_vec[i]).T)).T]), color = color[i+3], width = 1.5)
            
        #Axis info

        for i in range(3):
            pl.add_lines(np.array([[0,0,0], basis[i]/500*self.earth_radius*1.3]), color = color[i], width = 1.5)
        
        pl.show()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    earth_pb2_grpc.add_SatracServicer_to_server(Satrac(), server)

    server.add_insecure_port('[::]:50051')

    server.start()
    server.wait_for_termination()

def lat_lon_at(satellite, ti, earth_radius):
    geo = satellite.at(ti)

    lat, lon = wgs84.latlon_of(geo)
    lat, lon = lat.arcminutes()/60, lon.arcminutes()/60
    height = wgs84.height_of(geo).km

    return lat_lon_rotation(lat, lon, np.array([earth_radius + height, 0, 0]).T)

def normalize(value):
    if value > 1:
        return 1
    elif value < -1:
        return -1
    
    return value

def lat_lon_rotation(lat, lon, vector):
    s = math.sin(lat * math.pi / 180)
    c = math.cos(lat * math.pi / 180)
    dcm_y = np.array([[c, 0, s],
                      [0, 1, 0],
                      [-s, 0, c]])

    s = math.sin(lon * math.pi / 180)
    c = math.cos(lon * math.pi / 180)
    dcm_z = np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])

    res = (dcm_z @ (dcm_y @ vector))

    return res.T

def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


if __name__ == '__main__':
    logging.basicConfig()
    serve()