# import unittest
# import cProfile
# import sys
# import networkx as nx
# import matplotlib.pyplot as plt
# from a1 import intercept

# class TestExamples(unittest.TestCase):

# 	def testExample1(self):
# 		roads = [(6,0,3,1), (6,7,4,3), (6,5,6,2), (5,7,10,5), (4,8,8,5), (5,4,8,2),
# 		(8,9,1,2), (7,8,1,3), (8,3,2,3), (1,10,5,4), (0,1,10,3), (10,2,7,2),
# 		(3,2,15,2), (9,3,2,2), (2,4,10,5)]
# 		stations = [(0,1), (5,1), (4,1), (3,1), (2,1), (1,1)]
# 		start = 6
# 		friendStart = 0

# 		expected_output = (7, 9, [6,7,8,3])
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# 	def testExample2(self):
# 		roads = [(0,1,35,3), (1,2,5,2), (2,0,35,4), (0,4,10,1), (4,1,22,2),
# 		(1,5,65,1), (5,2,70,1), (2,3,10,1), (3,0,20,3)]
# 		stations = [(4,3), (5,2), (3,4)]
# 		start = 0
# 		friendStart = 4
# 		expected_output = None
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# 	def testExample3(self):

# 		roads = [(0,1,35,7), (1,2,5,4), (2,0,35,6), (0,4,10,5), (4,1,22,3),
# 		(1,5,60,4), (5,3,70,2), (3,0,10,7)]
# 		stations = [(4,2), (5,1), (3,4)]
# 		start = 0
# 		friendStart = 3
# 		expected_output = (160, 39, [0,1,2,0,1,2,0,4])
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# 	def testExample4(self):

# 		roads = [(0,1,10,7), (0,2,10,3), (2,0,1,4), (1,0,1,7)]
# 		stations = [(2,4), (1,3)]
# 		start = 0
# 		friendStart = 1
# 		expected_output = (10, 3, [0,2])
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# # Group 1: TestInterceptEdgeCases
# class TestExtra(unittest.TestCase):
# 	def test_never(self):
# 		roads = [(0, 1, 1, 1), (1, 0, 1, 1), (2, 1, 5, 1)]
# 		stations = [(0, 1), (1, 1)]
# 		start = 2
# 		friendStart = 0
# 		expected_output = (5, 1, [2, 1])
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# 	def test_gonna(self):
# 		roads = [(0, 1, 35, 3), (1, 0, 1, 1), (2, 0, 10, 5)]
# 		stations = [(0, 3), (1, 2)]
# 		start = 2
# 		friendStart = 0
# 		expected_output = (10, 5, [2, 0])
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# 	def test_give(self):
# 		roads = [(0, 1, 1, 2), (1, 2, 1, 3), (2, 0, 1, 5), (3, 0, 10, 10)]
# 		stations = [(0, 2), (1, 3), (2, 5)]
# 		start = 3
# 		friendStart = 0
# 		expected_output = (10, 10, [3, 0])
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# 	def test_you(self):
# 		roads = [(0, 1, 1, 1), (1, 0, 5, 2)]
# 		stations = [(0, 1), (1, 1)]
# 		start = 1
# 		friendStart = 0
# 		expected_output = (5, 2, [1, 0])
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# 	def test_up(self):
# 		roads = [(0, 1, 1, 2), (1, 2, 1, 3), (2, 0, 1, 5), (3, 2, 10, 5), (3, 0, 5, 1)]
# 		stations = [(0, 2), (1, 3), (2, 5)]
# 		start = 3
# 		friendStart = 0
# 		expected_output = (10, 5, [3, 2])
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# 	def test_never_(self):
# 		roads = [(0, 1, 1, 1), (1, 2, 1, 2), (2, 0, 1, 3), (3, 2, 10, 3), (3, 0, 5, 1)]
# 		stations = [(0, 1), (1, 2), (2, 3)]
# 		start = 3
# 		friendStart = 0
# 		expected_output = (10, 3, [3, 2])
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# 	def test_gonna_(self):
# 		roads = [(3, 2, 5, 2), (0, 1, 1, 1), (1, 2, 1, 1), (2, 0, 1, 1), (3, 0, 10, 1)]
# 		stations = [(0, 1), (1, 1), (2, 1)]
# 		start = 3
# 		friendStart = 0
# 		expected_output = (5, 2, [3, 2])
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# 	def test_let_(self):
# 		roads = [(0, 1, 3, 1), (1, 2, 3, 1), (2, 0, 3, 1), (0, 3, 10, 5), (2, 4, 2, 1), (3, 4, 1, 1), (4, 0, 5, 2)]
# 		stations = [(3, 2), (4, 2)]
# 		start = 0
# 		friendStart = 3
# 		expected_output = (11, 6, [0, 3, 4])
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# 	def test_you_(self):
# 		roads = [(0, 1, 1, 1), (1, 2, 1, 1), (2, 0, 5, 2), (1, 0, 2, 1), (2, 1, 2, 1)]
# 		stations = [(1, 2), (2, 3)]
# 		start = 0
# 		friendStart = 1
# 		expected_output = (2, 2, [0, 1, 2])
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# 	def test_down_(self):
# 		roads = [(0, 1, 3, 2), (1, 2, 3, 2), (2, 3, 3, 2), (0, 4, 6, 3), (4, 5, 1, 1), (5, 3, 1, 1), (3, 0, 10, 2)]
# 		stations = [(3, 2), (5, 1)]
# 		start = 0
# 		friendStart = 5
# 		expected_output = (26, 12, [0, 1, 2, 3, 0, 4, 5])
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# 	def test_never_(self):
# 		roads = [(0, 1, 10, 2), (0, 2, 100, 1), (1, 2, 5, 1), (2, 3, 10, 2), (1, 3, 50, 5), (3, 4, 2, 1), (2, 4, 30, 2), (4, 0, 5, 2)]
# 		stations = [(3, 2), (4, 2)]
# 		start = 0
# 		friendStart = 3
# 		expected_output = (27, 6, [0, 1, 2, 3, 4])
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# 	def test_1(self):
# 		roads = [(0, 2, 10, 3), (1, 2, 5, 2), (2, 1, 15, 5), (2, 0, 12, 10)]
# 		stations = [(0, 5), (1, 5)]
# 		start = 2
# 		friendStart = 0
# 		expected_output = (12, 10, [2, 0])
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# 	def test_2(self):
# 		stations_input = [(i, 2) for i in range(20)]
# 		start_input = 20
# 		friendStart_input = 0
# 		roads_input = [(20, 2, 7, 4), (20, 18, 30, 36)] + \
# 					  [(i, (i+1)%20, 1, 2) for i in range(20)]
# 		expected_output = (7, 4, [20, 2])
# 		result = intercept(roads_input, stations_input, start_input, friendStart_input)
# 		self.assertEqual(result, expected_output)

# 	def test_3(self):
# 		stations_input = [(i, 5) for i in range(20)]
# 		start_input = 20
# 		friendStart_input = 0
# 		roads_input = [(20, 5, 30, 25), (20, 0, 40, 100)] + \
# 					  [(i, (i+1)%20, 1, 1) for i in range(20)]
# 		expected_output = (30, 25, [20, 5])
# 		result = intercept(roads_input, stations_input, start_input, friendStart_input)
# 		self.assertEqual(result, expected_output)

# 	def test_4(self):
# 		roads = [(0, 1, 1, 1), (1, 0, 1, 1),
# 				 (2, 3, 1, 1), (3, 2, 1, 1)]
# 		stations = [(0, 5), (1, 5)]
# 		start = 2
# 		friendStart = 0
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertIsNone(result)

# 	def test_5(self):
# 		roads = [(0, 1, 10, 3), (1, 2, 20, 10), (2, 0, 8, 4)]
# 		stations = [(0, 5), (1, 5), (2, 5)]
# 		start = 1
# 		friendStart = 0
# 		expected_output = (20, 10, [1, 2])
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# 	def test_linear_path(self):
# 		roads = [(0,1,5,2), (1,2,5,2), (2,3,5,2), (3,4,5,2), (1,4,30,2), (4,0,5,2)]
# 		stations = [(4,2), (2,2)]
# 		start = 0
# 		friend_start = 4
# 		expected_output = (20, 8, [0,1,2,3,4])
# 		self.assertEqual(intercept(roads, stations, start, friend_start), expected_output)

# 	def test_long_chase(self):
		
# 		roads = [(i, i+1, 3, 5) for i in range(19)] + [(19, 0, 3, 4)]
# 		stations = [(i, 5) for i in range(20)]
# 		start = 0
# 		friend_start = 19

# 		#1900 edges traversed/95 cycles
# 		result = intercept(roads, stations, start, friend_start)
# 		self.assertEqual(result, (5700, 9405, [i for i in range(20)]*95+[0]))

# 	def test_dijkstra_tie(self):
# 		roads = [(6,0,3,1), (6,7,4,3), (6,5,6,2), (5,7,10,5), (4,8,8,5), (5,4,8,2),
# 		(8,9,1,2), (7,8,1,3), (8,3,2,3), (1,10,5,4), (0,1,10,3), (10,2,7,2),
# 		(3,2,15,2), (9,3,2,2), (2,4,10,5)]
# 		stations = [(0,1), (5,1), (4,1), (3,1), (2,1), (1,1)]
# 		start = 6
# 		friendStart = 0

# 		expected_output = (7, 9, [6,7,8,3])
# 		result = intercept(roads, stations, start, friendStart)
# 		self.assertEqual(result, expected_output)

# 	def test_double_road(self):
# 		roads = [(0, 1, 5, 5), (0, 1, 5, 5), (1, 2, 5, 5), (2, 0, 5, 5)]
# 		stations = [(2, 5), (1, 5)]
# 		start = 0
# 		friend_start = 2
		
# 		self.assertEqual(intercept(roads, stations, start, friend_start), (5, 5, [0,1]))

# 	def test_double_road_all(self):
# 		roads = [(0, 1, 5, 5), (0, 1, 5, 5), (0, 1, 6, 5), (0, 1, 5, 6), (1, 2, 5, 5), (2, 0, 5, 5)]
# 		stations = [(2, 5), (1, 5)]
# 		start = 0
# 		friend_start = 2
		
# 		self.assertEqual(intercept(roads, stations, start, friend_start), (5, 5, [0,1]))

# 	def test_short_none(self):
# 		roads = [(0, 1, 2, 1), (1, 0, 2, 1)]
# 		stations = [(0, 5), (1, 5)]
# 		start = 0
# 		friend_start = 1

# 		self.assertIsNone(intercept(roads, stations, start, friend_start))

# 	def test_short_long(self):
# 		roads = [(0, 1, 2, 1), (1, 0, 2, 1)]
# 		stations = [(0, 4), (1, 5)]
# 		start = 0
# 		friend_start = 1

# 		self.assertEqual(intercept(roads, stations, start, friend_start), (18, 9, [0,1]*5))

# 	def test_hi(self):
# 		mult = 1000000000000
# 		roads = [(0,1,9223372036854775807,7), (0,2,10,3 + 7 * mult), (2,0,1,4), (1,0,1,7)]
# 		stations = [(2,4), (1,3)]
# 		start = 0
# 		friendStart = 1

# 		self.assertEqual(intercept(roads, stations, start, friendStart), (10, 3 + 7 * mult, [0,2]))

# 	def test_hi2(self):
# 		mult = 1000000000000
# 		roads = [(0,1,10,7 + 7 * 5000), (0,2,10,3 + 7 * mult), (2,0,1,4), (1,0,1,7)]
# 		stations = [(2,4), (1,3)]
# 		start = 0
# 		friendStart = 1

# 		self.assertEqual(intercept(roads, stations, start, friendStart), (10, 7 + 7 * 5000, [0,1]))


# # Run all tests
# if __name__ == '__main__':
#     unittest.main()

import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from a1 import intercept

def test_random(
	iterationsPerGroup,
	minLocations,
	maxLocations,
	maxRoads,
	minRoadCost,
	maxRoadCost,
	minRoadTime,
	maxRoadTime,
	seed = 42,
):

	if maxRoads < maxLocations:
		return

	testOutput = {
		"data": {
			"minLocations": minLocations,
			"maxLocations": maxLocations,
			"iterationsPerGroup": iterationsPerGroup,
			"maxRoads": maxRoads,
			"minRoadCost": minRoadCost,
			"maxRoadCost": maxRoadCost,
			"minRoadTime": minRoadTime,
			"maxRoadTime": maxRoadTime
		},
		"out": []
	}

	random.seed(seed)

	for nLocations in range(minLocations, maxLocations + 1):

		locations = [i for i in range(nLocations)]

		for nStations in range(2, min(21, nLocations + 1)):

			group = {
				"nLocations": nLocations,
				"nStations": nStations,
				"tests": []
			}

			# for each number of locations and stations, we run n iterations.
			for i in range(iterationsPerGroup):
				
				# create train loop
				stations = random.sample(locations, nStations)
				stations = [(s, random.randint(1, 5)) for s in stations]

				# select friendStart
				friendStart = random.choice(stations)

				# create between location and maxRoads roads
				# ensure each location has at least 1 outgoing
				roads = [
					(
						location,
						random.choice([loc for loc in locations if loc != location]),
						random.randint(minRoadCost, maxRoadCost),
						random.randint(minRoadTime, maxRoadTime)
					)
				for location in locations]

				nExtraRoads = random.randint(0, maxRoads - len(roads))

				for i in range(nExtraRoads):
					src = random.choice(locations)
					dest = random.choice([loc for loc in locations if loc != src])
					
					roads.append((src, dest, random.randint(minRoadCost, maxRoadCost), random.randint(minRoadTime, maxRoadTime)))
				
				# select start
				start = random.choice(locations)

				result = intercept(roads, stations, start, friendStart)

				group['tests'].append({
					"testNo": i,
					"roads": roads,
					"stations": stations,
					"start": start,
					"friendStart": friendStart,
					"result": result
				})

			testOutput["out"].append(group)

	with open("out.json", "w") as f:
		json.dump(testOutput, f)

def test_random_parallel(
	iterationsPerGroup,
	minLocations,
	maxLocations,
	maxRoads,
	minRoadCost,
	maxRoadCost,
	minRoadTime,
	maxRoadTime,
	seed = 42,
):

	if maxRoads < maxLocations:
		return

	testOutput = {
		"data": {
			"minLocations": minLocations,
			"maxLocations": maxLocations,
			"iterationsPerGroup": iterationsPerGroup,
			"maxRoads": maxRoads,
			"minRoadCost": minRoadCost,
			"maxRoadCost": maxRoadCost,
			"minRoadTime": minRoadTime,
			"maxRoadTime": maxRoadTime,
			"totalTests": 0
		},
		"out": []
	}

	total_tests = 0
	total_tests_lock = threading.Lock()

	

	def run_test_group(nLocations, nStations):
		random.seed(seed)
		nonlocal total_tests

		group = {
				"nLocations": nLocations,
				"nStations": nStations,
				"tests": []
			}

		# for each number of locations and stations, we run n iterations.
		for i in range(iterationsPerGroup):
			
			# create train loop
			locations = [i for i in range(nLocations)]
			stations = random.sample(locations, nStations)
			stations = [(s, random.randint(1, 5)) for s in stations]

			# select friendStart
			friendStart = random.choice(stations)

			# create between location and maxRoads roads
			# ensure each location has at least 1 outgoing
			roads = [
				(
					location,
					random.choice([loc for loc in locations if loc != location]),
					random.randint(minRoadCost, maxRoadCost),
					random.randint(minRoadTime, maxRoadTime)
				)
			for location in locations]

			nExtraRoads = random.randint(0, maxRoads - len(roads))

			for i in range(nExtraRoads):
				src = random.choice(locations)
				dest = random.choice([loc for loc in locations if loc != src])
				
				roads.append((src, dest, random.randint(minRoadCost, maxRoadCost), random.randint(minRoadTime, maxRoadTime)))
			
			# select start
			start = random.choice(locations)

			result = intercept(roads, stations, start, friendStart)

			group['tests'].append({
				"roads": roads,
				"stations": stations,
				"start": start,
				"friendStart": friendStart,
				"result": result
			})

		with total_tests_lock:
			total_tests += iterationsPerGroup

		return group

	with ThreadPoolExecutor() as executor:
		futures = []

		for nLocations in range(minLocations, maxLocations + 1):
			for nStations in range(2, min(21, nLocations + 1)):
				futures.append(executor.submit(run_test_group, nLocations, nStations))
			
		for future in as_completed(futures):
			testOutput["out"].append(future.result())

	testOutput["data"]["totalTests"] = total_tests

	with open("out.json", "w") as f:
		json.dump(testOutput, f)

if __name__ == "__main__":

	test_random_parallel(
		iterationsPerGroup = 1,
		minLocations = 2,
		maxLocations = 100,
		maxRoads = 1000,
		minRoadCost = 1,
		maxRoadCost = 1000,
		minRoadTime = 1,
		maxRoadTime = 1000,
		seed = 42,
	)