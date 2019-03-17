






var net = require('net');


client.on('data', function(data) {
	console.log('Received: ' + data);
	client.destroy(); // kill client after server's response
});

//client.on('close', function() {
//	console.log('Connection closed');
//});



