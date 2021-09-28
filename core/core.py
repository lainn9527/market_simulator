from datetime import datetime, timedelta
from queue import Queue
from typing import Dict, List
from numpy.lib.function_base import angle

# from core import agent
from .market import Market, CallMarket
from .order import LimitOrder, MarketOrder
from .agent_manager import AgentManager


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
        config: Dict,
        market_type: str = "continuous",
        agents_type: str = "normal"
    ) -> None:

        # initialize all things
        self.message_queue = Queue()
        self.start_time = None
        self.timestep = None
        self.random_seed = None
        self.show_price = config['Core']['show_price']
        self.order_capacity = config['Market']['Structure']['order_capacity']
        self.agent_manager = AgentManager(self, config['Agent'])


        if market_type == "continuous":
            self.market = Market(self,
                                 interest_rate=config['Market']['Structure']['interest_rate'],
                                 interest_period=config['Market']['Structure']['interest_period'],
                                 clear_period=config['Market']['Structure']['clear_period'],
                                 transaction_rate=config['Market']['Structure']['transaction_rate'],
                                 securities=config['Market']['Securities'])
        
        elif market_type == "call":
            self.market = CallMarket(self,
                                     interest_rate=config['Market']['Structure']['interest_rate'],
                                     interest_period=config['Market']['Structure']['interest_period'],
                                     clear_period=config['Market']['Structure']['clear_period'],
                                     transaction_rate=config['Market']['Structure']['transaction_rate'],
                                     securities=config['Market']['Securities'])


    def run(self, num_simulation=100, num_of_timesteps=100000, random_seed=9527):
        # time
        self.start_time = datetime.now()
        self.random_seed = random_seed
        # register the core for the market and agents
        self.market.start()
        self.agent_manager.start(self.market.get_securities())

        print("Set up the following agents:")

        # start to simulate
        for i in range(num_simulation):
            self.timestep = 0
            self.market.open_session()
            for timestep in range(num_of_timesteps):
                self.step()

        return self.market.orderbooks, self.agent_manager

    def step(self):
        self.agent_manager.step()
        self.handle_messages()
        self.market.step()
        self.handle_messages()
        self.timestep += 1

        if self.show_price:
            print(
                f"At: {self.timestep}, the market state is:\n{self.market.market_stats()}\n")
            if self.timestep % 100 == 0:
                print(f"==========={self.timestep}===========\n")

    def env_start(self, random_seed):
        self.start_time = datetime.now()
        self.random_seed = random_seed
        self.market.start()
        self.agent_manager.start(self.market.get_securities())

        self.timestep = 0
        self.market.open_session()
        self.agent_manager.add_rl_agent(self.config['Env']['agent'])

        
    def env_step(self, action, rl_agent_id):
        self.agent_manager.env_step(action, rl_agent_id)
        self.handle_messages()
        self.market.step()
        self.timestep += 1
        # print(f"At: {self.timestep}, the market state is:\n{self.market.market_stats()}\n")
        # if self.timestep % 100 == 0:
        #     print(f"==========={self.timestep}===========\n")

    def env_close(self):
        return self.market.orderbooks, self.agent_manager

    def parallel_env_start(self, random_seed, group_name):
        self.start_time = datetime.now()
        self.random_seed = random_seed
        self.market.start()
        self.agent_manager.start(self.market.get_securities())
        pr_agent_ids = list(self.agent_manager.agents.keys())
        self.timestep = 0
        self.market.open_session()

        return pr_agent_ids


    def parallel_env_step(self, actions):
        self.agent_manager.multi_env_step(actions)
        self.handle_messages()
        self.market.step()
        self.timestep += 1
        # print(f"At: {self.timestep}, the market state is:\n{self.market.market_stats()}\n")
        # if self.timestep % 100 == 0:
        #     print(f"==========={self.timestep}===========\n")

    def parallel_env_close(self):
        return self.market.orderbooks, self.agent_manager

    def multi_env_start(self, random_seed, group_name):
        self.start_time = datetime.now()
        self.random_seed = random_seed
        self.market.start()
        self.agent_manager.start(self.market.get_securities())
        
        agent_ids = []
        for name in group_name:
            agent_ids += list(self.agent_manager.group_agent[name])
        self.timestep = 0
        self.market.open_session()

        return agent_ids


    def multi_env_step(self, actions): 
        self.agent_manager.multi_env_step(actions)
        self.handle_messages()
        done = self.market.step()
        self.timestep += 1
        
        return done

    def multi_env_close(self):
        return self.market.orderbooks, self.agent_manager

    def show_market_state(self):
        return self.market.market_stats()

    def send_message(self, message):
        # add message to queue
        self.message_queue.put(message)

    def announce_message(self, message):
        # announce to all agents immediately
        print(
            f"==========Time: {self.timestep}, {message.subject} Start==========")
        self.handle_message(message)

    def handle_messages(self):
        market_messages = 0
        agent_messages = 0
        while not self.message_queue.empty():
            message = self.message_queue.get()
            self.handle_message(message)
            if message.postcode == 'MARKET':
                market_messages += 1
            else:
                agent_messages += 1
        # print(f'Market messages: {market_messages}, agent messages: {agent_messages}')

    def handle_message(self, message):
        if message.postcode == 'MARKET':
            if message.receiver == 'market':
                self.market.receive_message(message)
            else:
                raise Exception(
                    'Postcode {message.postcode} and receiver {message.receiver} don\'t match')
        elif message.postcode == 'AGENT' or message.postcode == 'ALL_AGENTS':
            # check if agent exist:
            self.agent_manager.receive_message(message)
        else:
            raise Exception

    def queue_full(self):
        return self.message_queue.qsize() > self.order_capacity

    def get_order_record(self, code, order_id):
        return self.market.get_order_record(code, order_id)

    def get_current_price(self, code):
        return self.market.get_current_price(code)

    def get_records(self, code, _type, step=1, from_last = True):
        return self.market.get_records(code, _type, step, from_last)

    def get_best_bids(self, code, number):
        return self.market.get_best_bids(code, number)

    def get_best_asks(self, code, number):
        return self.market.get_best_asks(code, number)

    def get_tick_size(self, code):
        return self.market.get_tick_size(code)

    def get_transaction_rate(self):
        return self.market.get_transaction_rate()
    
    def get_risk_free_rate(self):
        return self.market.get_risk_free_rate()

    def get_stock_size(self):
        return self.market.stock_size

    def get_value(self, code):
        return self.market.get_value(code)
    
    def get_env_state(self, lookback, best_price):
        state = {'average': self.get_records(code='TSMC', _type = 'average', step = lookback),
                 'close': self.get_records(code='TSMC', _type = 'close', step = lookback),
                 'best_bid': self.get_records(code='TSMC', _type = 'bid', step = lookback),
                 'best_ask': self.get_records(code='TSMC', _type = 'ask', step = lookback),
                 'bids': self.get_best_bids(code='TSMC', number = best_price),
                 'asks': self.get_best_asks(code='TSMC', number = best_price),}
        return state

    def get_call_env_state(self, lookback, from_last = True):
        state = {
            'price': self.get_records(code='TSMC', _type = 'price', step = lookback, from_last = from_last),
            'volume': self.get_records(code='TSMC', _type = 'volume', step = lookback, from_last = from_last),
            'value': self.get_records(code='TSMC', _type = 'value', step = lookback, from_last = from_last),
            'risk_free_rate': self.get_risk_free_rate()
        }
        return state
    
    def get_rl_agent_status(self, agent_id):
        return self.agent_manager.agents[agent_id].action_status

    def get_base_price(self, code):
        return self.market.get_base_price(code)

    def get_base_volume(self, code):
        return self.market.get_base_volume(code)