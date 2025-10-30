from scgraph.geographs.north_america_rail import north_america_rail_geograph
from scgraph_data.world_railways import world_railways_geograph
from scgraph import Graph
# rail_output = north_america_rail_geograph.get_shortest_path(
# origin_node ={ 'latitude' : 34.1 , 'longitude': -118.2} ,
# destination_node ={ 'latitude' : 40.7 , 'longitude': -74.0} ,
# output_units = 'mi')

# print ('LA-NY North Americal Rail: Length :', rail_output ['length'])

# rail_output = world_railways_geograph.get_shortest_path(
# origin_node ={ 'latitude' : 34.1 , 'longitude': -118.2} ,
# destination_node ={ 'latitude' : 40.7 , 'longitude': -74.0} ,
# output_units = 'mi')

# print ('LA-NY GLobal Railway: Length :', rail_output ['length'])
# print('Path: ', rail_output['coordinate_path'])

# rail_output = world_railways_geograph.get_shortest_path(
# origin_node ={ 'latitude' : -34.91 , 'longitude': 138.58} ,
# destination_node ={ 'latitude' : -12.49 , 'longitude': 130.88} ,
# output_units = 'km')
# print ('The Ghan: Length:', rail_output ['length'])

# rail_output = world_railways_geograph.get_shortest_path(
# origin_node ={ 'latitude' : -34.91 , 'longitude': 138.58} ,
# destination_node ={ 'latitude' : -12.49 , 'longitude': 130.88} ,
# algorithm_fn = Graph.a_star ,
# algorithm_kwargs ={'heuristic_fn' : world_railways_geograph.haversine } ,
# output_units = 'km')
# print ('The Ghan: Length (A*):', rail_output ['length'])

graph = [
{1: 5 , 2: 1} ,
{0: 5 , 2: 2 , 3: 1} ,
{0: 1 , 1: 2 , 3: 4 , 4: 8} ,
{1: 1 , 2: 4 , 4: 3 , 5: 6} ,
{2: 8 , 3: 3} ,
{3: 6}
]
Graph.validate_graph ( graph = graph )
output = Graph.dijkstra ( graph = graph , origin_id =0 , destination_id =5)
print (output)

output = Graph.dijkstra_makowski ( graph = graph , origin_id =0 , destination_id =5)
print (output)

output = Graph.a_star ( graph = graph , origin_id =0 , destination_id =5, heuristic_fn=world_railways_geograph.haversine)
print (output)