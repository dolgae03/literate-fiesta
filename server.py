from concurrent import futures
from locale import normalize
import logging
from tkinter import X
from xml.dom.pulldom import END_DOCUMENT

from skyfield.api import load, wgs84

import grpc
import earth_pb2
import earth_pb2_grpc

import pyvista as pv
import math
import numpy as np

from astropy.coordinates import SkyCoord, GCRS, FK5 
from astropy.time import Time


# 서비스 이름으로 클래스를 생성하고, 서비스이름+{Servicer}의 클래스를 상속받습니다.
class Satrac(earth_pb2_grpc.SatracServicer):
    # .proto에서 지정한 메서드를 구현하는데, request, context를 인자로 받습니다.
    # 요청하는 데이터를 활용하기 위해서는 request.{메시지 형식 이름} 으로 호출합니다.
    # 응답시에는 메서드 return에 proto buffer 형태로 메시지 형식에 내용을 적어서 반환합니다.
    def __init__(self):
        self.station_url = 'https://celestrak.org/NORAD/elements/gp.php?INTDES=2022-072'
        self.view_degree = 0
        self.earth_radius = 6378.1
        self.sat_vec = []
        
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
            start_program(self.new_sat_trac, self.gs_location, self.now_time, self.sat_location,self.sat_dcm, self.sat_vec, self.sat_trac_communication)
        except Exception as e:
            return earth_pb2.Response_code(message = 'The Error Occured while starting program : %s' % e)

        return earth_pb2.Response_code(message = 'Success : StartDraw')

    def ChangeUrl(self, request, context):
        self.station_url = request.url
        
        return earth_pb2.Response_code(message = 'Success : ChangeUrl')
    
    def SendGroundStationInfo(self, request, context):
        self.gs_location = z_rotation(np.array([request.x,request.y,request.z]), self.rot_ang)
        self.view_degree = request.degree

        if self.Trajectoryflag == False:
            return earth_pb2.Response_code(message = 'The Error Occured : %s' % ('Satinfoflag is diabled'))

        geo = self.satellite.at(self.now_time)
        x, y, z = geo.position.km
        lat,lon = wgs84.latlon_of(geo)

        lat, lon = lat.dms()[0] + (lat.dms()[0]/60) + (lat.dms()[2]/3600), lon.dms()[0] + (lon.dms()[0]/60) + (lon.dms()[2]/3600)
        print(lat,lon)
        lat, lon = math.asin(normalize(z/self.earth_radius)), math.atan2(y, x)
        print(lat, lon)

        lat, lon = math.asin(normalize(self.gs_location[2]/self.earth_radius)), math.atan2(self.gs_location[1], self.gs_location[0])
        crisp = wgs84.latlon(lat, lon)
        
        t_deg, events_deg = self.satellite.find_events(crisp, self.now_time, self.end_time, altitude_degrees=self.view_degree)

        two_time_set = []

        aos_flag = False
        for ti, event in zip(t_deg, events_deg):
            if event == 0:
                taos = ti
                print(ti.utc_strftime('%Y %b %d %H:%M:%S'),  end='')
                aos_flag = True
            elif event == 2 and aos_flag == True:

                two_time_set.append((taos, ti))
                aos_flag = False

        if aos_flag == True:
            two_time_set.append((taos, t_deg[-1]))

        print('the interval : %d' % len(two_time_set))
        
        time_gap = 1/(24*60*60)
        self.sat_trac_communication = []

        for start_utc, end_utc in two_time_set:
            li = []
            while end_utc - start_utc > 0:
                start_utc += time_gap
                geo = self.satellite.at(start_utc)
                li.append(np.array(geo.position.km))

            xc, yc, zc = zip(*li)
            new_cord = SkyCoord(x = xc, y = yc, z = zc, frame = GCRS(), unit = 'kpc' , representation_type = 'cartesian').transform_to(FK5(equinox = Time(self.now_time.tdb, format = 'jd')))
            new_cord.representation_type = 'cartesian'

            self.sat_trac_communication.append(np.array(list(zip(new_cord.x.value, new_cord.y.value, new_cord.z.value))))


        self.GroundStationinfoflag= True

        return earth_pb2.Response_code(message = 'Success : SendGroundStationInfo')
    
    def SendTrajectoryInfo(self, request, context):
        ts = load.timescale()
        self.now_time = ts.utc(request.t.y, request.t.mo, request.t.d, request.t.h, request.t.m, request.t.s)
        sat_num = request.sat_num
        self.rot_ang = calculate_rotation(self.now_time)

        try:
            satellites = load.tle_file(self.station_url, reload=True)
            self.satellite = {sat.model.satnum: sat for sat in satellites}[sat_num]

        except Exception as e :
            return earth_pb2.Response_code(message = 'The Error Occured : %s' % e)

        print(self.satellite)

        time_gap = 1/(24*60*60)
        sat_trac = []

        self.end_time = self.now_time

        for i in range(10000):
            self.end_time += time_gap
            geo = self.satellite.at(self.end_time)
            sat_trac.append(np.array(geo.position.km))

        xc, yc, zc = zip(*sat_trac)
        new_cord = SkyCoord(x = xc, y = yc, z = zc, frame = GCRS(), unit = 'kpc' , representation_type = 'cartesian').transform_to(FK5(equinox = Time(self.now_time.tdb, format = 'jd')))
        new_cord.representation_type = 'cartesian'

        self.new_sat_trac = np.array(list(zip(new_cord.x.value, new_cord.y.value, new_cord.z.value)))
        self.sat_location = self.new_sat_trac[0]

        self.Trajectoryflag = True
        return earth_pb2.Response_code(message = 'Success : SendTrajectoryInfo')

    def SendAttitudeInfo(self,request,context):
        quaternion = np.array([request.q1, request.q2, request.q3, request.q4]).T
        
        self.sat_dcm = q_t_d(quaternion)
        
        self.Attitudeinfoflag = True
        return earth_pb2.Response_code(message = 'Success : SendAttitudeInfo')

    def SendSatVecInfo(self,request_iterator,context):
        
        for request in request_iterator:
            self.sat_vec.append(np.array([request.u, request.v, request.w]).T)

        return earth_pb2.Response_code(message = 'Success : SendOtherVecInfo')

def serve():
	# 서버를 정의할 때, futures의 멀티 스레딩을 이용하여 서버를 가동할 수 있습니다.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # 위에서 정의한 서버를 지정해 줍니다.
    earth_pb2_grpc.add_SatracServicer_to_server(Satrac(), server)
    
    # 불안정한 포트 50051로 연결합니다.
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

def normalize(value):
    if value > 1:
        return 1
    elif value < -1:
        return -1
    
    return value

def calculate_rotation(now_time):
    ts = load.timescale()
    
    ref_juli_solar = ts.utc(tuple(now_time.utc)[0], 3, 20, 12, 0, 0)
    ref_juli_earth = ts.utc(tuple(now_time.utc)[0],tuple(now_time.utc)[1], tuple(now_time.utc)[2], 12, 0, 0)
    solar_rot = (now_time.tdb - ref_juli_solar.tdb)/365*360
    earth_rot = (now_time.tdb - ref_juli_earth.tdb)*360

    print(solar_rot, earth_rot, solar_rot-earth_rot)

    return solar_rot-earth_rot


def z_rotation(vector, angle):
    
    s = math.sin(angle * math.pi / 180)
    c = math.cos(angle * math.pi / 180)

    dcm = np.array([[c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1]])
    
    return (dcm @ vector.T).T

def q_t_d(li):
    q1, q2, q3, q4 = li

    return np.array([[(q4**2 + q1**2 -q2**2-q3**2), 2*(q1*q2+q3*q4), 2*(q1*q3-q2*q4)],
                     [2*(q1*q2-q3*q4), (q4**2-q1**2 + q2**2 - q3**2), 2*(q2*q3 + q1*q4)],
                     [2*(q1*q3+q2*q4), 2*(q2*q3 - q1*q4), (q4**2 - q1**2 - q2**2 + q3**2)]]) 

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def start_program(sat_trac, ground_station, now_time, sat_location, sat_att_dcm, sat_vec, sat_trac_communication):
    earth_radius = 6378.1
    rot_ang = calculate_rotation(now_time)

    sphere = pv.Sphere(radius=earth_radius, theta_resolution=240, phi_resolution=240, start_theta=270.0001, end_theta=270)
    sphere.active_t_coords = np.zeros((sphere.points.shape[0], 2))


    for i in range(sphere.points.shape[0]):
        x, y, z = sphere.points[i, 0]/earth_radius, sphere.points[i, 1]/earth_radius, sphere.points[i, 2]/earth_radius
        x, y, z = normalize(x), normalize(y), normalize(z)
        
        sphere.active_t_coords[i] = [0.5 + math.atan2(-x, y)/(2 * math.pi), 0.5 + math.asin(z)/math.pi]

    #cordinate chage
    
    sphere.rotate_z(-90+rot_ang)

    #load satellite

    pl = pv.Plotter()
    pl.add_background_image('starry-night-sky-fit.jpg', scale=1.001)
    pl.add_mesh(sphere, texture=pv.read_texture("earth.jpg"), smooth_shading=False)
    
    x, y, z = sat_location

    pl.add_lines(sat_trac, width = 0.5, color = 'white')

    for each_trac in sat_trac_communication:
        pl.add_lines(np.array(each_trac), width = 10, color = 'yellow')
    #pl.add_points(ground_station, render_points_as_spheres=True, point_size = 15, color = 'white')
    
    basis = np.array([[500,0,0],
                      [0,500,0],
                      [0,0,500]])
    sat_att = (sat_att_dcm @ basis).T
    color = ['red','blue','green','yellow','purple','orange', 'white', 'skyblue']

    reader = pv.get_reader('./library/satellite.stl')
    sat_mesh = reader.read()
    sat_scale = 30

    reader = pv.get_reader('./library/antenna.stl')
    gs_mesh = reader.read()
    gs_scale = 2
    gs_dcm = rotation_matrix_from_vectors(np.array([0,1,0]), np.array(ground_station))    

    # 0, 1, 0

    for i in range(gs_mesh.points.shape[0]):
        gs_mesh.points[i] = (gs_dcm @ (gs_mesh.points[i].T)).T
        x, y, z = gs_mesh.points[i,0]*gs_scale + ground_station[0], gs_mesh.points[i,1]*gs_scale + ground_station[1], gs_mesh.points[i,2]*gs_scale + ground_station[2]
        gs_mesh.points[i,0], gs_mesh.points[i,1], gs_mesh.points[i,2] = x, y, z

    for i in range(sat_mesh.points.shape[0]):
        sat_mesh.points[i] = (sat_att_dcm @ (sat_mesh.points[i].T)).T
        x, y, z = sat_mesh.points[i,0]*sat_scale + sat_location[0], sat_mesh.points[i,1]*sat_scale + sat_location[1], sat_mesh.points[i,2]*sat_scale + sat_location[2]
        sat_mesh.points[i,0], sat_mesh.points[i,1], sat_mesh.points[i,2] = x, y, z 
 
    pl.add_mesh(gs_mesh)
    pl.add_mesh(sat_mesh)
    for i in range(3):
        pl.add_lines(np.array([sat_location + sat_att[i],sat_location]), color = color[i], width = 1.5)
    
    #Some vector

    print(sat_vec)

    for i in range(len(sat_vec)):
        print(i)
        pl.add_lines(np.array([sat_location , sat_location + 500*(sat_att_dcm@((sat_vec[i]).T)).T]), color = color[i+3], width = 1.5)
        
    #Axis info

    for i in range(3):
        pl.add_lines(np.array([[0,0,0], basis[i]/500*earth_radius*1.3]), color = color[i], width = 1.5)
    
    pl.show()


if __name__ == '__main__':
    logging.basicConfig()
    serve()