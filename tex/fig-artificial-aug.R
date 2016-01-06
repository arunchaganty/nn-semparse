# Make plot about artificial data
pdf('fig-artificial-aug.pdf', width=5, height=5, pagecentre=T)
par(family='serif')
x = c(0, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500)
y_entities = c(6.2, 29.4, 20.8, 54.8, 58.4, 61.4, 66.6, 72.8, 68.8, 74.0, 77.4)
# y_nesting = c(6.2, 73.8, 84.8, 78.6, 87.4, 84.8, 88.8, 93.4, 68.8, 95.4, 80.4)
y_both = c(6.2, 82.0, 83.8, 72.0, 92.2, 89.6, 93.2, 98.8, 93.6, 95.2, 97.4)
y_more_depth2 = c(6.2, 28.6, 74.6, 87.2, 89.0, 99.6, 100, 99.8, 100, 100, 100)
y_more_depth4 = c(6.2, 84.8, 95.2, 98.8, 100, 100, 100, 99.4, 100, 100, 100)

#y_min = min(c(min(y_entities), min(y_nesting), min(y_both)))
y_min = 0
plot(x, y_more_depth2, col='black', ylim=c(y_min - 2, 100), type='l',
     xlab='Number of additional examples',
     ylab='Accuracy (%)',
     #cex=1.5, cex.axis=1.5, cex.lab=1.5) 
     )
lines(x, y_more_depth4, col='red', lty='dashed')
lines(x, y_entities, col='green3', lty='dotdash')
#lines(x, y_nesting, col='blue', lty='dotted')
lines(x, y_both, col='blue', lty='twodash')
legend('bottomright', 
       c('same-domain, independent', 'out-of-domain, independent',
         'same domain, correlated', 'out-of-domain, correlated'),
       col=c('black', 'red', 'green3', 'blue'),
       lty=c('solid', 'dashed', 'dotdash', 'twodash'))
dev.off()
