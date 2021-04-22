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
        self.real_start_time = None
        self.simulated_time = None
        self.randomizer = None
        self.agent_info = {agent_type: len(list_of_agents) for agent_type, list_of_agents in agents.items()}
        self.agents = {agent.unique_id: agent for list_of_agents in agents.values() for agent in list_of_agents}
                
        self.market = market
    
    def run(self, num_simulation = 100, num_of_days = 1, time_scale = 0.001):
        # time
        self.real_start_time = datetime.now()
        self.simulated_time = datetime.fromisoformat('2021-03-22 08:30:00.000')
        
        # register the core for the market and agents
        self.market.start(self, self.simulated_time, time_scale)
        for _, agent in self.agents.items():
            agent.start(self, self.simulated_time, time_scale, self.market.get_securities())
        print("Set up the following agents:")
        for agent_type, num in self.agent_info.items():
            print(f"   {agent_type}: {num}")

        # start to simulate
        open_auction_timestep = int(1800 * pow(time_scale, -1) - 1) # 0800-0900, final order is at 08:59:59.999
        continuous_trading_timestep = int(15900 * pow(time_scale, -1) - 1) # 0900-1325, final order is at 13:24:59.999
        close_auction_timestep = int(300 * pow(time_scale, -1) - 1) # 1325-1330, final order is at 13:29:59.999
        total_timestep = open_auction_timestep + continuous_trading_timestep + close_auction_timestep
        for day in range(num_of_days):
            # use the core to control the state of market, not itself, for efficiency (probably)
            # open session at 08:30:00 and start the auction
            self.market.open_session(self.simulated_time, total_timestep)
            self.market.start_auction(open_auction_timestep)
            for timestep in range(open_auction_timestep):
                self.step()
                self.simulated_time += timedelta(seconds = time_scale)

            # close auction at 08:59:59.999
            self.simulated_time += timedelta(seconds = time_scale)
            self.market.close_auction()
            self.handle_messages()

            # continuous trading at 09:00:00.000
            self.market.start_continuous_trading(continuous_trading_timestep)
            for timestep in range(continuous_trading_timestep):
                self.step()
                self.simulated_time += timedelta(seconds = time_scale)

            # close continuous trading at 13:24:59.999
            self.simulated_time += timedelta(seconds = time_scale)
            self.market.close_continuous_trading()

            # open auction at 13:25:00.000
            self.market.start_auction(close_auction_timestep)
            for timestep in range(close_auction_timestep):
                self.step()
                self.simulated_time += timedelta(seconds = time_scale)
            self.simulated_time += timedelta(seconds = time_scale)
            self.market.close_auction()
            self.handle_messages()
            self.market.close_session()

            # update the time
            self.simulated_time += timedelta(days = 1)
            self.simulated_time.replace(hour = 8, minute = 30)
    
    def step(self):
        # agents make desicion
        for _, agent in self.agents.items():
            agent.step()

        # check the message queue and execute the actions from agents on the market
        self.handle_messages()
        
        self.market.step() # what to do?
        # add timestep
        if self.simulated_time.second == 0 and self.simulated_time.microsecond == 0:
            print(f"At: {self.simulated_time.isoformat()}, the trading amount is: {self.market.market_stats()}")




    def send_message(self, message, send_time):
        # check valid (what valid)

        # add message to queue
        self.message_queue.put(message)

    def announce(self, message, send_time):
        # announce to all agents immediately
        print(f"==========Time: {self.simulated_time}, {message.subject} Start==========")
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
            if self.check_agent(message.receiver):
                self.agents[message.receiver].receive_message(message)
            else:
                raise Exception('The agent: {message.receiver} doesn\'t exist')
            
        elif message.postcode == 'ALL_AGENTS':
            for _, agent in self.agents.items():
                agent.receive_message(message)

        else:
            raise Exception
    
    def check_agent(self, agent):
        return True

    def best_bids(self, code, number):
        return self.market.best_bids(code, number)

    def best_asks(self, code, number):
        return self.market.best_asks(code, number)

    def get_price_info(self, code):
        return self.market.get_price_info(code)