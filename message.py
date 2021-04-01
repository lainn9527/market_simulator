class Message:
    def __init__(self, postcode, subject, sender, receiver, content):
        self.postcode = postcode
        self.subject = subject
        self.sender = sender
        self.receiver = receiver
        self.content = content
    
    @property
    def valid_subject(self):
        return ['OPEN_SESSION', 'CLOSE_SESSION', 'MARKET_ORDER', 'LIMIT_ORDER', 'AUCTION_ORDER', 'CANCEL_ORDER', 'MODIFY_ORDER', 'OPEN_AUCTION', 'CLOSE_AUCTION', 'OPEN_CONTINUOUS_TRADING', 'STOP_CONTINUOUS_TRADING', 'ORDER_PLACED', 'ORDER_CANCELLED', 'ORDER_INVALIDED', 'ORDER_FILLED', 'ORDER_FINISHED']
    
    @property
    def postcodes(self):
        return ['ALL_AGENTS', 'AGENT', 'MARKET']