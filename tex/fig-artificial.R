# Make plot about artificial data
do_plot <- function(title, y_simple, y_nested, y_union) {
  y_min = min(c(min(y_simple), min(y_nested), min(y_union)))
  x = c(0, 25, 50, 75, 100, 150, 200, 250, 300)
  plot(x, y_simple, col='black', ylim=c(y_min - 2, 100), type='l',
       main=title, 
       xlab='Number of additional training examples',
       ylab='Acuracy (%)',
       cex=1.5, cex.axis=1.5, cex.lab=1.5) 
  lines(x, y_nested, col='blue')
  lines(x, y_union, col='red')
  legend('bottomright', 
         c('Add Simple examples', 'Add Nested examples', 'Add Union examples'),
         col=c('black', 'blue', 'red'), lty=1, cex=1.5)
}
pdf('fig-artificial.pdf', width=12, height=4, pagecentre=T)
par(mfrow=c(1,3), family='serif')

# Eval on Simple
y_simple = c(73.2, 86.8, 94.4, 99.8, 98.2, 100, 100, 100, 100)
y_nested = c(73.2, 71.8, 94.4, 88.0, 95.8, 96.2, 99.6, 100, 100)
y_union = c(73.2, 60.4, 91.2, 94.4, 93.0, 98.6, 96.4, 100, 99.6)
do_plot('Simple Domain (black)', y_simple, y_nested, y_union)

# Eval on Nested
y_simple = c(43.4, 42.8, 59.2, 81.0, 86.8, 97.4, 97.0, 99.6, 98.2)
y_nested = c(43.4, 75.6, 63.8, 89.8, 91.6, 99.2, 99.8, 100, 99.4)
y_union = c(43.4, 39.2, 89.2, 92.0, 96.4, 97.6, 96.4, 94.4, 99.0)
do_plot('Nested Domain (blue)', y_simple, y_nested, y_union)

# Eval on Union
y_simple = c(40.0, 25.2, 85.4, 87.2, 90.6, 94.4, 95.4, 98.0, 100)
y_nested = c(40.0, 83.8, 81.2, 91.8, 93.8, 98.4, 99.5, 100, 97.6)
y_union = c(40.0, 68.8, 97.8, 99.4, 99.4, 99.8, 99.6, 100, 100)
do_plot('Union Domain (red)', y_simple, y_nested, y_union)

dev.off()
