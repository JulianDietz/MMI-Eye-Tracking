using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;


public class DisplayAOIs : MonoBehaviour
{
    public AOI[] allAOIs;
    public string filename;

    string fixJson(string value)
    {
        value = "{\"Items\":" + value + "}";
        return value;
    }


    // Start is called before the first frame update
    void Start()
    {
        //string json_file = "aoi_config.json";
        string json_raw = "hallo";
        using (StreamReader r = new StreamReader(filename))
        {
            json_raw = r.ReadToEnd();
        }
        string jsonString = fixJson(json_raw);
        allAOIs = JsonHelper.FromJson<AOI>(jsonString);
        Debug.Log(allAOIs[0].aoi_id);
        Debug.Log(allAOIs[1].aoi_id);

        for (int i = 0; i < allAOIs.Length; i++)
        {
            GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
            print(allAOIs[i].get_width());
            print(allAOIs[i].get_height());
            print(allAOIs[i].get_d());


            //cube.transform.position = new Vector3(allAOIs[i].get_x(), allAOIs[i].get_y(), allAOIs[i].get_z());
            //cube.transform.localScale = new Vector3(allAOIs[i].get_width(), allAOIs[i].get_height(), allAOIs[i].get_d());
            //Gizmos.DrawCube(new Vector3(allAOIs[i].center_x, allAOIs[i].center_y, allAOIs[i].center_z), new Vector3(allAOIs[i].width, allAOIs[i].height, allAOIs[i].dev));
        }
    }

    void OnDrawGizmos()
    {
        for (int i = 0; i < allAOIs.Length; i++)
        {
            Color myColor = new Color();
            ColorUtility.TryParseHtmlString(allAOIs[i].color, out myColor);
            Gizmos.color = myColor;
            Gizmos.DrawCube(new Vector3(allAOIs[i].get_center_x(), allAOIs[i].get_center_y(), allAOIs[i].get_center_z()*-1), new Vector3(allAOIs[i].get_width(), allAOIs[i].get_height(), allAOIs[i].get_d()));
        }
    }
}
