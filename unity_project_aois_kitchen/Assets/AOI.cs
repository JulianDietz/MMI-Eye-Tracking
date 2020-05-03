using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using System;


[System.Serializable]
public class AOI
{
    public int aoi_id;
    public int cross_hair_x;
    public int cross_hair_y;
    public int cross_hair_z;
    public int aoi_x_0;
    public int aoi_y_0;
    public int aoi_z_0;
    public int aoi_x_1;
    public int aoi_y_1;
    public int aoi_z_1;
    public int aoi_x_2;
    public int aoi_y_2;
    public int aoi_z_2;
    public string color;
    public string name;




    //public int center_x = x + (w/2);
    //public int center_y = y + (h/2);
    //public int center_z = z + (z/2);


    public int get_x()
    {
        return Math.Min(Math.Min(aoi_x_0, aoi_x_1), aoi_x_2);
    }

    public int get_y()
    {
        return Math.Min(Math.Min(aoi_y_0, aoi_y_1), aoi_y_2);
    }

    public int get_z()
    {
        return Math.Min(Math.Min(aoi_z_0, aoi_z_1), aoi_z_2);
    }

    public int get_width()
    {
        return Math.Max(Math.Max(aoi_x_0, aoi_x_1), aoi_x_2) - get_x();
    }

    public int get_height()
    {
        return Math.Max(Math.Max(aoi_y_0, aoi_y_1), aoi_y_2) - get_y();
    }

    public int get_d()
    {
        return Math.Max(Math.Max(aoi_z_0, aoi_z_1), aoi_z_2) - get_z();
    }

    public int get_center_x()
    {
        return Math.Min(Math.Min(aoi_x_0, aoi_x_1), aoi_x_2) + (get_width()/2);
    }

    public int get_center_y()
    {
        return Math.Min(Math.Min(aoi_y_0, aoi_y_1), aoi_y_2) + (get_height()/2);
    }

    public int get_center_z()
    {
        return Math.Min(Math.Min(aoi_z_0, aoi_z_1), aoi_z_2) + (get_d()/2);
    }


}
