const express = require('express');
const app = express();
const port = 3000;
let {PythonShell} = require('python-shell')


app.get('/', (req, res) => res.send('Hello World!'))

app.get('/predict', callPredict)
function callPredict(req, res) {
  var options = {
    args:
    [
      req.query.funds, // starting funds
      req.query.size, // (initial) wager size
      req.query.count, // wager count = number of wagers per sim
      req.query.sims // number of simulations
    ]
  }

    var spawn = require('child_process').spawn;
    var process = spawn('python3', ['./main.py',
				   JSON.stringify(req.query)]);
    
    process.stdout.on('data', function (data) {
	res.send(data.toString());
    });

    process.stderr.on('data', function (data) {
	res.send(data.toString());
    });

}


app.listen(port, () => console.log(`Example app listening on port ${port}!`))
