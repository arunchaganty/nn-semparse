# Make plot about artificial data
pdf('fig-artificial-aug.pdf', width=6, height=6, pagecentre=T)
par(family='serif')
x = c(0, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500)
y_entities = c(40.6, 65.8, 76.6, 56.8, 79.8, 82.0, 92.0, 88.4, 89.6, 92.8, 91.6)
y_nesting = c(40.6, 80.6, 20.9, 92.4, 80.8, 95.8, 95.0, 79.0, 95.8, 99.8, 82.0)
y_both = c(40.6, 89.6, 85.6, 99.6, 99.6, 94.2, 100, 94.8, 99.4, 100, 100)

y_min = min(c(min(y_entities), min(y_nesting), min(y_both)))
plot(x, y_entities, col='black', ylim=c(y_min - 2, 100), type='l',
     xlab='Number of synthetic examples',
     ylab='Accuracy (%)',
     cex=1.5, cex.axis=1.5, cex.lab=1.5) 
lines(x, y_nesting, col='blue')
lines(x, y_both, col='red')
  legend('bottomright', 
         c('Entity-based', 'Nesting-based', 'Both'),
         col=c('black', 'blue', 'red'), lty=1, cex=1.5)
dev.off()
