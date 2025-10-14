
const tailwindConf = require('../WebGasket/tailwind.config.js')

tailwindConf.content = [
      "./src/*.jsx",
      "../WebGasket/js/components/**/*.jsx",
      "../WebGasket/js/components/*.jsx",
      "../WebGasket/js/widgets/*.jsx"
]

module.exports = tailwindConf