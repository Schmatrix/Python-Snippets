from basic_face_detection_v01 import df #executes face_recog.py script and imoports df dataframe
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource

df["Start_string"]=df["Start_Time"].dt.strftime("%Y-%m-%d %H:%M:%S") #formating data into a string for tooltip display
df["End_string"]=df["End_Time"].dt.strftime("%Y-%m-%d %H:%M:%S") #formating data into a string for tooltip display


p=figure(x_axis_type="datetime", height=100, width=500, sizing_mode="scale_width", title="Motion Graph")
p.yaxis.minor_tick_line_color = None
p.toolbar.logo = None


q=p.quad(source=ColumnDataSource(df), left="Start_Time", right="End_Time", bottom=0, top=1, color="Green")
p.add_tools(HoverTool(tooltips=[("Start: ","@Start_string"),("End: ","@End_string")]))


output_file("Graph.html")
show(p)
