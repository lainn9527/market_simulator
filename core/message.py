class Message:
    def __init__(self, postcode, subject, sender, receiver, content):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self._subject = subject
        self._valid_subject = ['OPEN_SESSION', 'CLOSE_SESSION', 'MARKET_ORDER', 'LIMIT_ORDER', 'AUCTION_ORDER', 'CANCELLATION_ORDER', 'MODIFICATION_ORDER', 'OPEN_AUCTION', 'CLOSE_AUCTION', 'OPEN_CONTINUOUS_TRADING', 'STOP_CONTINUOUS_TRADING', 'ORDER_PLACED', 'ORDER_CANCELLED', 'ORDER_INVALIDED', 'ORDER_FILLED', 'ORDER_FINISHED']
        self._postcode = postcode
        self._valid_postcode = ['ALL_AGENTS', 'AGENT', 'MARKET']

    @property
    def subject(self):
        return self._subject

    @subject.setter
    def subject(self, value):
        if value not in self._valid_subject:
            raise Exception
        self._subject = value
    
    @property
    def postcode(self):
        return self._postcode
    
    @postcode.setter
    def postcode(self, value):
        if value not in self._valid_postcode:
            raise Exception
        self._postcode = value


    def __str__(self):
        return f"{self.sender}-> {self.receiver}: {self.subject}, {self.content}"
