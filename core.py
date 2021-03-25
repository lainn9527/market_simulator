from datetime import datetime, timedelta
from queue import Queue
from order import LimitOrder, MarketOrder
from agent import Agent
from market import Market
class Core:
    '''
        Serve as platform to contain the market and agents.
        Function:
            1. Message passing
            2. Maintain the global time
            3. Maintain the best bid & ask price
            4. Parallel in # of simulations
    '''
    def __init__(
        self,
        market: Market,
        agents: [agent]
    ) -> None:
    
        # initialize all things
        self.message_queue = Queue()
        self.real_start_time = None
        self.simulated_time = None
        self.randomizer = None
        self.agents = agents
        self.market = market
    
    def run(self, num_simulation = 100, time_scale = 0.001):
        # time
        self.real_start_time = datetime.now()
        # in ms
        self.simulated_time = datetime.fromisoformat('2021-03-22 09:00:00.000')
        
        # start market
        self.market.start(self, datetime.now())

        # start agent
        for agent in self.agents:
            agent.start(self, datetime.now())

        # start to simulate
        total_timestep = 16200 * pow(time_scale, -1)
        for timestep in range(total_timestep):
            step()
            self.simulated_time += timedelta(seconds = time_scale)
    
    def step(self):
        # check every agent
        for agent in self.agents:
            agent.step()
        # check the message queue
        for msg in self.message_queue:
            self.handle_message(msg)
        
        # check the market
        # market publish new price
        market.step()

    def send_message(self, message, send_time):
        # check valid (what valid)

        # add message to queue
        self.message_queue.put(message)

    def market_best_bids(self, code):
        return self.market.best_bids(code)
    
    def market_best_asks(self, code):
        return self.market.best_asks(code)

    def handle_message(self, message):
        while self.message_queue.empty() is False:
            msg = self.message_queue.get()
            if msg.postcode == 'MARKET':
                if msg.receiver == 'market':
                    self.market.receive_message(msg)
                else:
                    raise Exception

            elif msg.postcode == 'AGENT':
                # check if agent exist:
                if check_agent(msg.receiver):
                    pass
                else:
                    raise Exception
            
            else:
                raise Exception

    def check_agent(self, agent_id):
        pass


class Message:
    def __init__(self, posecode, subject, sender, receiver, content):
        self.postcode = postcode
        self.subject = subject
        self.sender = sender
        self.receiver = receiver
        self.content = content
    
    @property
    def message_subject(self):
        return ['MARKET_ORDER', 'LIMIT_ORDER', 'CANCEL_ORDER', 'MODIFY_ORDER', 'MARKET_OPEN', 'MARKET_CLOSE', 'ORDER_CONFIRMED', 'ORDER_EXECUTED', 'ORDER_CANCELLED']

