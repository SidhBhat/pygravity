import matplotlib
import matplotlib.pyplot as pyplot
import numpy
from matplotlib.animation import FuncAnimation

G = numpy.double(1);

# Execption Classes
class Zero_MassERROR(Exception):
	pass;

class Body:
	def _set_vector(vec_like):
		vec = numpy.array(vec_like, dtype=numpy.double);
		numpy.reshape(vec,2)
		return vec;

	def __init__(self,
			name     = "Body",
			color    = "k",
			mass     = 1,
			position = [0, 0]
			):

		if(not mass):
			raise Zero_MassERROR("The mass of a body cannot be zero");
		if (not (type(name) is str and type(color) is str)):
			raise ValueError("\'name\' nad \'color\' expect string arguments only")

		self.position = Body._set_vector(position);
		self.mass     = numpy.double(mass);
		self.name     = name;
		self.color    = color;

	def __str__(self):
		return format({
			'name'     : self.name,
			'color'    : self.color,
			'mass'     : self.mass,
			'position' : self.position
			});

	__repr__ = __str__;

	def __getitem__(self, key):
		if (type(key) is str):
			if (key == 'name'):
				return self.name;
			elif (key == 'color'):
				return self.color;
			elif (key == 'mass'):
				return self.mass;
			elif (key == 'position'):
				return self.position;
			else:
				raise IndexError("Invalid Key value");
		else:
			raise TypeError("Key must be a string value");

	def __setitem__(self, key, value):
		if (type(key) is str):
			if (key == 'name'):
				if (type(value) is str):
					self.name = value;
				else:
					raise TypeError("Body['name'] must be a string value");
			elif (key == 'color'):
				if (type(value) is str):
					self.color = value;
				else:
					raise TypeError("Body['color'] must be a string value");
			elif (key == 'mass'):
				if (value):
					self.mass = numpy.double(value);
				else:
					raise Zero_MassERROR("The mass of a body cannot be zero");
			elif (key == 'position'):
				self.position     = Body._set_vector(value);
			else:
				raise IndexError("Invalid Key value");
		else:
			raise TypeError("Key must be a string value");

class Dynamic_Body(Body):
	def __init__(self,
			name         = "Body",
			color        = "k",
			mass         = 1,
			position     = [0, 0],
			velocity     = [0, 0],
			acceleration = [0, 0]
			):

		Body.__init__(self, name, color, mass, position);
		self.velocity     = Body._set_vector(velocity);
		self.acceleration = Body._set_vector(acceleration);

	_half = numpy.double(0.5);

	def update_velocity(self, tDelta):
		self.velocity += self.acceleration * tDelta;

	def update_position(self, tDelta):
		self.position += \
			(self.velocity + self._half * self.acceleration * tDelta) * tDelta;

	def __str__(self):
		return format({
			'name'         : self.name,
			'color'        : self.color,
			'mass'         : self.mass,
			'position'     : self.position,
			'velocity'     : self.velocity,
			'acceleration' : self.acceleration
			});

	__repr__ = __str__;

	def __getitem__(self, key):
		if (type(key) is str):
			if (key == 'name'):
				return self.name;
			elif (key == 'color'):
				return self.color;
			elif (key == 'mass'):
				return self.mass;
			elif (key == 'position'):
				return self.position;
			elif (key == 'velocity'):
				return self.velocity;
			elif (key == 'acceleration'):
				return self.acceleration;
			else:
				raise IndexError("Invalid Key value");
		else:
			raise TypeError("Key must be a string value");

	def __setitem__(self, key, value):
		if (type(key) is str):
			if (key == 'name'):
				if (type(value) is str):
					self.name = value;
				else:
					raise TypeError("Body['name'] must be a string value");
			elif (key == 'color'):
				if (type(value) is str):
					self.color = value;
				else:
					raise TypeError("Body['color'] must be a string value");
			elif (key == 'mass'):
				if (value):
					self.mass = numpy.double(value);
				else:
					raise Zero_MassERROR("The mass of a body cannot be zero");
			elif (key == 'position'):
				self.position     = Body._set_vector(value);
			elif (key == 'velocity'):
				self.velocity     = Body._set_vector(value);
			elif (key == 'acceleration'):
				self.acceleration = Body._set_vector(value);
			else:
				raise IndexError("Invalid Key value");
		else:
			raise TypeError("Key must be a string value");

class Body_list:
	"""argument to __init__ 'body_list' is a list of dictionaries of the form:
	{
	'name':     "<name>",   # a string (optional)
	'color':    '<color>',  # single character (optional)
	'mass':     <mass>,
	'position': [x_pos, y_pos],
	'velocity': [x_vel, y_vel],
	}
	"""
	def __init__(self,
			*body_list # Expects a list of dictionaries
			):
		self.body_list = [];
		n = int(1);
		for body in body_list:
			self.body_list.append(Dynamic_Body(
				name         = body.get('name', f"Body {n}"),
				color        = body.get('color', 'k'),
				mass         = body.get('mass', 1),
				position     = body.get('position', [0, 0]),
				velocity     = body.get('velocity', [0, 0])
				));
			n += 1;

	# with syntax add_body(mass=1, position=[1,0], ...)
	def add_body(self, **body):
		self.body_list.append(Dynamic_Body(
				name         = body.get('name', f"Body {len(self.body_list) + 1}"),
				color        = body.get('color', 'k'),
				mass         = body.get('mass', 1),
				position     = body.get('position', [0, 0]),
				velocity     = body.get('velocity', [0, 0])
				));

	# specify index or name of body
	def remove_body(self, body_id):
		if (type(body_id) is str):
			index_list = [];
			n = int(0);
			for body in self.body_list:
				if (body.name == body_id):
					index_list.append(n);
				n += 1;
			if (not len(index_list)):
				raise ValueError(f"Body with name \'{body_id}\' not found");
			index_list.sort(reverse=True);
			for index in index_list:
				self.body_list.pop(index);
		elif (type(body_id) is int):
			self.body_list.pop(body_id);
		else:
			raise IndexError(f"Specifed body_id \'{body_id}\' not found");

	def __len__(self):
		return len(self.body_list);

	def __getitem__(self, index):
		if (not type(index) is int):
			raise TypeError("Indices must be integers only");
		elif (index >= len(self)):
			raise IndexError("Index out of range");
		return self.body_list[index];

	class _Iterator:
		def __init__(self, body_list_obj):
			self._data = body_list_obj;
			self._pos = int(0);
		def __next__(self):
			if (self._pos >= len(self._data)):
				raise StopIteration;
			val = self._data[self._pos];
			self._pos += int(1);
			return val;

	def __iter__(self):
		return self._Iterator(self);

class Environment(Body_list):
	"""argument to __init__ 'body_list' is a list of dictionaries of the form:
	{
	'name':     "<name>",   # a string (optional)
	'color':    '<color>',  # single character (optional)
	'mass':     <mass>,
	'position': [x_pos, y_pos],
	'velocity': [x_vel, y_vel],
	}
	"""
	def __init__(self,
			*body_list # Expects a list of dictionaries
			):
		if (len(body_list)):
			Body_list.__init__(self, body_list);
		else:
			Body_list.__init__(self);

		# initialise data members
		self.figure, self.axes = (None, None);
		self.plot_properties = self._Plot_properties();


	def min_mass(self):
		return numpy.double(min(body.mass for body in self.body_list));

	class _Plot_properties:
		#default values
		def __init__(self):
			# the timestep interval
			self.timestep     = numpy.double(0.0001);
			self.markersize   = numpy.double(20);
			self.fps          = numpy.double(20);
			self.xbound       = numpy.double(5);
			self.ybound       = numpy.double(5);
			self.trace_length = numpy.double(100);

	def draw(self, **properties):
		self.figure, self.axes = pyplot.subplots(1,1);
		self.set_simulation_opts(**properties);
		min_mass = self.min_mass();

		compute = _Compute(self);
		compute.populate_acceleration_data();
		compute.setup_bodies();
		compute.plot_setup();
		compute.plot_bodies();

		# set axis properties
		# self.axes.set_title('Current Position');
		self.axes.axis('square');
		self.axes.autoscale(False);
		self.axes.axis('off');
		self.axes.set_xlim(-self.get_xbound(),self.get_xbound());
		self.axes.set_ylim(-self.get_xbound(),self.get_xbound());

		pyplot.show();

	# alias functions
	show = draw;

	def simulate(self, time, **properties):
		self.set_simulation_opts(**properties);
		self.figure, self.axes = pyplot.subplots(1,1);
		compute   = _Compute(self);
		fps       = self.get_fps();
		timestep  = self.get_timestep();
		time_text = None;
		time_template = 'time = %.1fs';

		def setup():
			nonlocal time_text, time_template;
			compute.populate_acceleration_data();
			compute.setup_bodies();
			compute.plot_setup();
			time_text = self.axes.text(0.05, 0.9, (time_template % 0.0),
							  transform = self.axes.transAxes);
			artists = compute.plot_bodies();
			artists.append(time_text);
			return artists;

		def animate(frame_data):
			for i in range(int(1 / (fps * timestep))):
				compute.update_bodies();
			time_text.set_text(time_template % (frame_data / fps));
			artists = compute.plot_bodies();
			artists.append(time_text);
			return artists;

		if(self.get_fps()*self.get_timestep() > numpy.double(1)):
			raise ValueError(f"timestep \'{timestep}\' is too large")

		ani = FuncAnimation(self.figure, animate, numpy.arange(1, fps*time, dtype=numpy.double),
			init_func=setup, interval=1000 / fps , repeat=False, blit=True);

		# set axis properties
		# self.axes.set_title('Simulation Window');
		self.axes.axis('square');
		self.axes.autoscale(False);
		self.axes.axis('off');
		self.axes.set_xlim(-self.get_xbound(),self.get_xbound());
		self.axes.set_ylim(-self.get_xbound(),self.get_xbound());

		pyplot.show();

	# call Signature:
	## set_simulation_opts(timestep=<timestep>, markersize=<markersize>,
	##       fps=<fps>, xbound=<xbound>, ybound=<ybound>, [,bounds=<bounds>],
	##       trace=<trace>)
	def set_simulation_opts(self, **opts):
		for key in opts.keys():
			if(type(key) is str):
				if(key == 'timestep' or key == 'tdelta' or key == 'ts'):
					self.set_timestep(opts[key]);
				elif(key == 'markersize' or key == 'ms'):
					self.set_markersize(opts[key]);
				elif(key == 'fps'):
					self.set_fps(opts[key]);
				elif(key == 'xbound' or key == 'xlim'):
					self.set_xbound(opts[key]);
				elif(key == 'ybound' or key == 'ylim'):
					self.set_xbound(opts[key]);
				elif(key == 'bounds' or key == 'lims'):
					self.set_bounds(opts[key]);
				elif(key == 'trace' or key == 'tr' or key == 'trace_length'):
					self.set_plot_trace(opts[key]);
				else:
					raise IndexError(f"Invalid Key Value \'{key}\'");
			else:
				raise TypeError("key must be of type string");

	def set_timestep(self, timestep):
		self.plot_properties.timestep = numpy.double(timestep);

	def set_markersize(self, size):
		self.plot_properties.markersize = numpy.double(size);

	def set_fps(self, fps):
		self.plot_properties.fps = numpy.double(fps);

	def set_xbound(self, lim):
		self.plot_properties.xbound = numpy.double(lim);

	def set_ybound(self, lim):
		self.plot_properties.ybound = numpy.double(lim);

	def set_bounds(self, lim):
		self.set_xbound(lim);
		self.set_ybound(lim);

	def set_plot_trace(self, trace):
		self.plot_properties.trace_length = numpy.double(trace);

	def get_timestep(self):
		return self.plot_properties.timestep

	def get_markersize(self):
		return self.plot_properties.markersize

	def get_fps(self):
		return self.plot_properties.fps

	def get_xbound(self):
		return self.plot_properties.xbound

	def get_ybound(self):
		return self.plot_properties.ybound

	def set_gravitational_constant(self, Num):
		global G;
		G = numpy.double(Num);

	def get_gravitational_constant(self):
		global G;
		return G;

# Helper class for computations
class _Compute(Environment):
	# perform a shallow copy of the Environment instance
	def __init__(self, Env):
		#initialise the Body_list members.
		self.body_list   = Env.body_list;

		# initialise Environment members
		self.figure, self.axes = Env.figure, Env.axes;
		self.plot_properties = Env.plot_properties;

		# Fix min mass for an instance
		self._min_mass = Env.min_mass();

		# redundant
		assert self.body_list is Env.body_list;

	# experimental value!!
	# for calcluateing the radius;
	A = numpy.double(12345);

	def every_other_body(self, body):
		n = int(0);

		while(n < len(self)):
			bd = self[n];
			if (not body is bd):
				yield bd;
			n += 1;

	def min_mass(self):
		return self._min_mass;

	def merge_two_bodies(self, bd1, bd2):
		if (bd1 is bd2):
			raise ValueError("Cannot merge the same body onto itself");

		delete_string = '__TO_BE_DELETED__';
		Mass     = bd1.mass + bd2.mass;
		velocity = (1 / Mass) * (bd1.mass * bd1.velocity + \
					bd2.mass * bd2.velocity);

		if (bd2.mass >= bd1.mass):
			bd2.mass = Mass;
			bd2.velocity = velocity;
			bd1.name = delete_string;
			bd = bd2;
		else:
			bd1.mass = Mass;
			bd1.velocity = velocity;
			bd2.name = delete_string;
			bd = bd1;
		self.remove_body(delete_string);

		return bd;

	def markersize(self, body):
		return (body.mass / self.min_mass()) ** (2 / 3) * self.get_markersize();

	def radius(self, body):
		return (body.mass / self.min_mass()) ** (1 / 3) * numpy.sqrt(self.get_markersize() / self.A);

	def two_bodies_overlap(self, bd1, bd2):
		norm   = numpy.linalg.norm;

		if (bd1 is bd2):
			raise ValueError("Cannot calculate overlap for the same body");

		if (norm(bd1.position - bd2.position) <= self.radius(bd1) + self.radius(bd2)):
			return True;
		return False;

	def change_intertial_frame(self, velocity):
		velocity = numpy.array(velocity, dtype=numpy.double);
		numpy.reshape(velocity,2);

		for body in self:
			body.velocity -= velocity;

	def zero_momentum_frame(self):
		Mass = numpy.double(0);
		velocity = numpy.zeros(2);

		for body in self:
			Mass += body.mass;
			velocity += body.mass * body.velocity;
		return (1/Mass) * velocity;

	# this routine will setup the bodies up for simulation, Specifically:
	## overlapping bodies are merged...
	## the intertial frame is set to the net zero momentum frame
	def setup_bodies(self):
		bd_list = self.body_list;
		for body in bd_list:
			for bd in self.every_other_body(body):
				if(self.two_bodies_overlap(body,bd)):
					self.merge_two_bodies(body,bd);
		self.change_intertial_frame(self.zero_momentum_frame());

	def populate_acceleration_data(self):
		global G;
		norm  = numpy.linalg.norm;

		for body in self.body_list:
			# assumes sum calles the appropriate numpy method!!
			body.acceleration = sum(
				(G * bd.mass / norm(bd.position - body.position) ** 3) * \
					(bd.position - body.position)
				for bd in self.every_other_body(body));

	def _collision_handler(self, body):
		for bd in self.every_other_body(body):
			if(self.two_bodies_overlap(body, bd)):
				self.merge_two_bodies(body,bd);
		self.change_intertial_frame(self.zero_momentum_frame());

	def update_bodies(self):
		global G;
		norm   = numpy.linalg.norm;
		power  = numpy.power;
		tDelta = self.get_timestep();

		for body in self:
			self._collision_handler(body);
		for body in self:
			# assumes sum calles the appropriate numpy method!!
			body.acceleration = sum(
				(G * bd.mass / power(norm(bd.position - body.position),3)) * \
					(bd.position - body.position)
				for bd in self.every_other_body(body));
			body.update_velocity(tDelta);
			body.update_position(tDelta);

	def plot_setup(self):
		for body in self:
			if (not hasattr(body,'trace')):
				body.xtrace = [];
				body.ytrace = [];
			body.trace, = self.axes.plot([],[], body.color+'.', lw=1, ms=1);

	def plot_bodies(self):
		artists = [];
		circles  = [];
		min_mass = self.min_mass();

		for body in self:
			if (len(body.xtrace) < self.plot_properties.trace_length):
				body.xtrace.append(body.position[0]);
				body.ytrace.append(body.position[1]);
			else:
				body.xtrace.pop(0);
				body.ytrace.pop(0);
				body.xtrace.append(body.position[0]);
				body.ytrace.append(body.position[1]);
			body.trace.set_data(body.xtrace, body.ytrace);
			circles.append(
				pyplot.Circle(body.position, radius=self.radius(body), color=body.color)
				)
			artists.append(body.trace);
		artists.append(self.axes.add_collection(
			matplotlib.collections.PatchCollection(circles, match_original=True)));

		return artists;

	## If you don't want the trace to show up in Environment.draw(), uncomment this
	# def __del__(self):
	# 		for body in self:
	# 			if(hasattr(body, 'trace')):
	# 				del body.trace;
	# 				del body.xtrace;
	# 				del body.ytrace;
