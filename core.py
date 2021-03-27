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
    
    def run(self, num_simulation = 100, num_of_days = 1, time_scale = 0.001):
        # time
        self.real_start_time = datetime.now()
        self.simulated_time = datetime.fromisoformat('2021-03-22 08:30:00.000')
        
        # register the core for the market and agents
        self.market.start(self, self.simulated_time)
        for agent in self.agents:
            agent.start(self, self.simulated_time)

        # start to simulate
        open_auction_timestep = (1800-1) * pow(time_scale, -1) # 0800-0900, final order is at 08:59:59.999
        continuous_trading_timestep = (15900-1) * pow(time_scale, -1) # 0900-1325, final order is at 13:24:59.999
        close_auction_timestep = (300-1) * pow(time_scale, -1) # 1325-1330, final order is at 13:29:59.999

        for day in range(num_of_days):
            # open session at 08:30:00 and start the auction
            self.market.open_session(self.simulated_time)
            for timestep in range(open_auction_timestep):
                self.step()
                self.simulated_time += timedelta(seconds = time_scale)
            # simulated_time == 08:59:59.999
            self.market.close_auction()

            # continuous trading
            for timestep in range(continuous_trading_timestep):
                self.step()
                self.simulated_time += timedelta(seconds = time_scale)

            # close market
            self.market.close_session()
            for timestep in range(close_auction_timestep):
                self.step()
                self.simulated_time += timedelta(seconds = time_scale)
            self.market.close_auction()

            # update the time
            self.simulated_time += timedelta(days = 1)
            self.simulated_time.replace(hour = 8, minute = 30)
    
    def step(self):
        # agents make desicion
        for agent in self.agents:
            agent.step()

        # check the message queue and execute the actions from agents on the market
        for msg in self.message_queue:
            self.handle_message(msg)

        
        
        market.step() # what to do?



    def send_message(self, message, send_time):
        # check valid (what valid)

        # add message to queue
        self.message_queue.put(message)

    def market_best_bids(self, code):
        return self.market.best_bids(code)
    
    def market_best_asks(self, code):
        return self.market.best_asks(code)

    def handle_message(self, message):
        if message.postcode == 'MARKET':
            if message.receiver == 'market':
                self.market.receive_message(message)
            else:
                raise Exception('Postcode {message.postcode} and receiver {message.receiver} don\'t match')

        elif message.postcode == 'AGENT':
            # check if agent exist:
            if check_agent(message.receiver):
                self.agents[message.receiver].receive_message(message)
            else:
                raise Exception('The agent: {message.receiver} doesn\'t exist')
            
        elif message.postcode == 'ALL_AGENTS':
            for agent in self.agents:
                agent.receive_message(message)

        else:
            raise Exception
    
    def check_agent(self, agent):
        pass



class Message:
    def __init__(self, postcode, subject, sender, receiver, content):
        self.postcode = postcode
        self.subject = subject
        self.sender = sender
        self.receiver = receiver
        self.content = content
    
    @property
    def subject(self):
        return ['MARKET_ORDER', 'LIMIT_ORDER', 'AUCTION_ORDER', 'CANCEL_ORDER', 'MODIFY_ORDER', 'MARKET_OPEN', 'OPEN_AUCTION', 'CLOSE_AUCTION', 'MARKET_CLOSE', 'ORDER_CONFIRMED', 'ORDER_EXECUTED', 'ORDER_CANCELLED']
    
    @property
    def postcodes(self):
        return ['ALL_AGENTS', 'AGENT', 'MARKET']

