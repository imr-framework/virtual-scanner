
function coms_sender(jsonData) {
var client = new net.Socket();
var TCPIP = '127.0.0.1';
var TCP_port = 1337;

client.connect(TCP_port, TCPIP, function() {
	console.log('Connected');
	client.write(jsonData);
});


client.on('close', function() {
	console.log('Connection closed');
	var status = 0;}


