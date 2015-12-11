default: jar

DEPS := $(shell ls lib/*.jar) $(shell find src -name "*.java") $(shell find src -name "*.scala")

classes: $(DEPS)
	mkdir -p classes
	javac -cp 'lib/*' -d classes `find src -name "*.java"`
	time scalac -cp 'lib/*' -d classes `find src -name "*.java"` `find src -name "*.scala"`

jar: classes
	mkdir -p evaluator
	jar cf evaluator/evaluator.jar -C classes .
	jar uf evaluator/evaluator.jar -C src .

clean:
	rm -rf classes evaluator/evaluator.jar
