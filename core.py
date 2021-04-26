import agent
import market
from datetime import datetime, timedelta
from queue import Queue
from order import LimitOrder, MarketOrder
from typing import Dict, List
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
        market: market.Market,
        agents: Dict[str, List]
    ) -> None:
    
        # initialize all things
        self.message_queue = Queue()
        self.start_time = None
        self.timestep = None
        self.random_seed = None
        self.agent_info = {agent_type: len(list_of_agents) for agent_type, list_of_agents in agents.items()}
        self.agents = {agent.unique_id: agent for list_of_agents in agents.values() for agent in list_of_agents}
        self.market = market
    
    def run(self, num_simulation = 100, num_of_timesteps = 100000, random_seed = 9527):
        # time
        self.start_time = datetime.now()
        self.random_seed = random_seed
        # register the core for the market and agents
        self.market.start(self)
        for agent in self.agents.values():
            agent.start(self, self.market.get_securities())

        print("Set up the following agents:")
        for agent_type, num in self.agent_info.items():
            print(f"   {agent_type}: {num}")

        # start to simulate
        for i in range(num_simulation):
            self.timestep = 0
            self.market.open_session()
            for timestep in range(num_of_timesteps):
                self.step()

        return self.market.orderbooks, self.agents
    
    def step(self):
        # agents make desicion
        # TODO: use agent manager!!!!!
        for agent in self.agents.values():
            agent.step()

        # check the message queue and execute the actions from agents on the market
        self.handle_messages()
        
        self.market.step() # what to do?
        # add timestep
        self.timestep += 1

        if self.timestep % 100 == 0:
            print(f"At: {self.timestep}, the market state is:\n{self.market.market_stats()}")


    def send_message(self, message):
        # check valid (what valid)

        # add message to queue
        self.message_queue.put(message)

    def announce_message(self, message):
        # announce to all agents immediately
        print(f"==========Time: {self.timestep}, {message.subject} Start==========")
        self.handle_message(message)

    def handle_messages(self):
        while not self.message_queue.empty():
            self.handle_message(self.message_queue.get())

    def handle_message(self, message):
        if message.postcode == 'MARKET':
            if message.receiver == 'market':
                self.market.receive_message(message)
            else:
                raise Exception('Postcode {message.postcode} and receiver {message.receiver} don\'t match')

        elif message.postcode == 'AGENT':
            # check if agent exist:
            self.agents[message.receiver].receive_message(message)
            
        elif message.postcode == 'ALL_AGENTS':
            for _, agent in self.agents.items():
                agent.receive_message(message)

        else:
            raise Exception
    
    def get_order_record(self, code, order_id):
        return self.market.get_order_record(code, order_id)

    def get_current_price(self, code):
        return self.market.get_current_price(code)

    def get_best_bids(self, code, number):
        return self.market.get_best_bids(code, number)

    def get_best_asks(self, code, number):
        return self.market.get_best_asks(code, number)

    def get_tick_size(self, code):
        return self.market.get_tick_size(code)

    def get_price_info(self, code):
        return self.market.get_price_info(code)
