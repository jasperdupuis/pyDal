# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 14:45:39 2021

@author: Jasper
"""

import numpy as np

import haversine

class Source():
    """
    A source can be a ship, point model, or something towed.
    
    Setting aside the source's spectrum, it will have
    course, in lat-lon tuples.
    depth, in m
    speed, in m/s for each tuple above
    name
    """
    
    def __init__(self):
        self.course = 'not set'
        self.depth = 'not set'
        self.speed = 'not set' #m/s
        self.name = 'point source'
        
    def set_depth(self,p_z = 4):
        self.depth = p_z
        
    def set_speed(self,p_v = 3):
        self.speed = p_v
        
    def set_name(self,p_name = 'Point source'):
        self.name = p_name
        
    def generate_course(self,
                        p_CPA_lat_lon,
                        p_CPA_deviation_m,
                        p_CPA_deviation_heading=haversine.Direction.EAST,
                        p_course_heading=0.45*np.pi, #(0,2pi), mathematical not navigation angles
                        p_distance=200,
                        p_divisions = 25):

        new_center = haversine.inverse_haversine(p_CPA_lat_lon, 
                                                 p_CPA_deviation_m, 
                                                 p_CPA_deviation_heading,
                                                 unit=haversine.Unit.METERS)
        # returns tuple (lat, lon)
        course_start = haversine.inverse_haversine(new_center,
                                                   p_distance//2,
                                                   p_course_heading,
                                                   unit=haversine.Unit.METERS)
        course_end = haversine.inverse_haversine(new_center,
                                                 p_distance//2,
                                                 p_course_heading+np.pi,
                                                 unit=haversine.Unit.METERS)
        
        lats = np.linspace(course_start[0],course_end[0],p_divisions)
        lons = np.linspace(course_start[1],course_end[1],p_divisions)
        course = []
        for la, lo, in zip(lats,lons):
            course.append((la,lo))
        #Leave as list of tuples, tuples are input to other things.
        self.course = course
        return course